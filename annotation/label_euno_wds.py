"""客户 GCS/本地 WDS 数据的 Gemini 增强标注(第一步作业)。

数据存放 = euno数据集说明.md: shard-*.tar 内 .pyd(16 帧 JPEG bytes),
支持 gs://bucket/path 与本地目录两种 --wds-dir。
与 gemini_labeler(视频文件版)的差异: 输入是**按时间顺序的 16 帧图**,
以 16 个 image part 喂 Gemini;prompt/校验/断点续跑逻辑复用。

产出(与 annotation_spec 3.2 对齐): 每行
  {"video_id", "gemini_model", "temperature", "gemini_output":
    {"attributes"(7 组), "reasoning_chain", "predictions"}}
后接一致性过滤(GT 已在 euno 标注中,走 gt 模式):
  python -m data.euno_wds --annotation <euno标注.json> --wds-dir <dir> \
      --out DATA/labels.jsonl                     # ① GT 转通用格式
  python -m annotation.label_euno_wds --wds-dir <dir> \
      --annotation <euno标注.json> --out pass1.jsonl \
      --model gemini-3.1-pro --vertex-project <P>  # ② 本脚本
  python -m annotation.consistency_filter --mode gt \
      --gemini pass1.jsonl --gt DATA/labels.jsonl --out-dir filtered/  # ③

用法要点:
  --shards 0-3        只处理部分分片(作业可按分片切分并行/多机)
  --annotation        只标注该标注文件覆盖的 key(如 balanced 100k
                      是 wds_full 1M 样本的子集,不传会标全量)
  --workers 8         并发标注(注意配额;单条 2.5-flash ~5s)
  断点续跑: --out 已有的 video_id 自动跳过,可随时中断重启
"""
import argparse
import io
import json
import os
import pickle
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annotation.gemini_labeler import build_prompt, validate_record  # noqa: E402

FRAMES_PREFACE = ("以下是同一段家用摄像头视频按时间顺序均匀抽取的 16 帧"
                  "(第 1 帧最早,第 16 帧最晚)。请把它们当作一段视频理解。\n")


def open_binary(path):
    """gs:// 或本地路径 → 可顺序读的二进制流。"""
    if path.startswith("gs://"):
        from google.cloud import storage
        bucket_name, blob_name = path[5:].split("/", 1)
        client = storage.Client()
        return client.bucket(bucket_name).blob(blob_name).open("rb")
    return open(path, "rb")


def read_json(path):
    with open_binary(path) as f:
        return json.load(io.TextIOWrapper(f, encoding="utf-8"))


def iter_shard_samples(wds_dir, shard_id):
    """顺序流式读一个 tar 分片,yield (sample_key, frames: List[bytes])。
    流式(r|)兼容 GCS blob——无需随机寻址,整分片一遍过。"""
    path = f"{wds_dir.rstrip('/')}/shard-{shard_id:06d}.tar"
    with open_binary(path) as raw, tarfile.open(fileobj=raw, mode="r|") as tf:
        for member in tf:
            if not member.name.endswith(".pyd"):
                continue
            data = pickle.load(tf.extractfile(member))
            yield data["video_rel"], data["frames"]


def label_frames_one(client, model_id, frames, prompt, temperature,
                     max_retries=3):
    """16 帧 JPEG bytes → Gemini。返回 (record, errs)。"""
    from google.genai import types
    contents = [FRAMES_PREFACE]
    contents += [types.Part.from_bytes(data=b, mime_type="image/jpeg")
                 for b in frames]
    contents.append(prompt)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id, contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=8192,      # thinking 预算坑,勿降
                    response_mime_type="application/json"))
            d = json.loads(resp.text)
            errs = validate_record(d)
            if not errs:
                return d, None
            if attempt == max_retries - 1:
                return d, errs
        except Exception as e:
            if attempt == max_retries - 1:
                return None, [f"{type(e).__name__}: {e}"]
            time.sleep(5 * (attempt + 1))
    return None, ["unreachable"]


def parse_shards(spec, index):
    if spec:
        if "-" in spec:
            lo, hi = spec.split("-")
            return list(range(int(lo), int(hi) + 1))
        return [int(x) for x in spec.split(",")]
    return sorted(set(index.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds-dir", required=True, help="gs://… 或本地目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotation", default=None,
                    help="euno 标注 json;只标其覆盖的 key(不传=标全量)")
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--shards", default=None, help="如 0-3 或 0,5,7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vertex-project", default=None)
    ap.add_argument("--location", default="global")
    a = ap.parse_args()

    from google import genai
    client = genai.Client(vertexai=True, project=a.vertex_project,
                          location=a.location) if a.vertex_project \
        else genai.Client()
    prompt = build_prompt()

    index = read_json(f"{a.wds_dir.rstrip('/')}/index.json")
    want = None
    if a.annotation:
        anns = read_json(a.annotation)
        want = {x["videos"][0] for x in anns}
        print(f"[scope] annotation 覆盖 {len(want)} keys")

    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["video_id"] for l in open(a.out)}
        print(f"[resume] {len(done)} already labeled")

    lock = threading.Lock()
    stats = {"ok": 0, "err": 0}
    fout = open(a.out, "a", encoding="utf-8")
    ferr = open(a.out + ".errors", "a", encoding="utf-8")

    def work(item):
        key, frames = item
        d, errs = label_frames_one(client, a.model, frames, prompt,
                                   a.temperature)
        with lock:
            if d is not None and not errs:
                fout.write(json.dumps(
                    {"video_id": key, "gemini_model": a.model,
                     "temperature": a.temperature, "gemini_output": d},
                    ensure_ascii=False) + "\n")
                fout.flush()
                stats["ok"] += 1
            else:
                ferr.write(json.dumps({"video_id": key, "errors": errs,
                                       "partial": d},
                                      ensure_ascii=False) + "\n")
                ferr.flush()
                stats["err"] += 1
            if (stats["ok"] + stats["err"]) % 50 == 0:
                print(f"ok={stats['ok']} err={stats['err']}", flush=True)

    def todo():
        n = 0
        for sid in parse_shards(a.shards, index):
            for key, frames in iter_shard_samples(a.wds_dir, sid):
                if key in done or (want is not None and key not in want):
                    continue
                if a.limit and n >= a.limit:
                    return
                n += 1
                yield key, frames

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(work, todo()))
    fout.close()
    ferr.close()
    print(f"done: ok={stats['ok']} err={stats['err']} -> {a.out}")


if __name__ == "__main__":
    main()
