"""无机位标识数据的伪 camera_id 生成(背景指纹聚类)。

背景: training_plan 2.1 的防泄漏切分需要 camera/household ID;
若客户数据无此字段,split_by_camera 会退化为按 clip 随机切
→ 模型"记住这个门廊",验证指标虚高。
方法: 固定机位的背景占画面绝大部分 → 首帧灰度缩略图(16×16,
归一化)即机位指纹;贪心阈值聚类得伪 camera_id,写回 labels.jsonl
的 meta.camera_id。

用法:
  python -m data.camera_fingerprint --labels DATA/labels.jsonl \
      --video-root videos/ [--threshold 0.15] [--out DATA/labels_cam.jsonl]
验证口径: 聚类数应接近真实机位数;同机位视频背景肉眼一致。
阈值调参: 簇数异常多 → 调大 threshold;异常少 → 调小。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fingerprint_from_frame(frame, size=16):
    """单帧(H,W,3)→ 灰度 16×16 → 零均值单位范数向量(光照鲁棒)。"""
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    v = g.astype(np.float32).flatten()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def fingerprint(rec_or_path, size=16):
    """首帧指纹。入参: 视频路径,或带 meta.storage=='wds' 的 record
    (euno 数据无视频文件,从 WDS 分片取首帧)。"""
    import cv2
    if isinstance(rec_or_path, dict):
        from data.euno_wds import load_wds_frames
        return fingerprint_from_frame(load_wds_frames(rec_or_path)[0], size)
    cap = cv2.VideoCapture(rec_or_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return fingerprint_from_frame(frame, size)


def cluster(fps, threshold=0.15):
    """贪心聚类: 与已有簇心距离(1-余弦)< threshold 归入,否则开新簇。
    返回每条的簇号。纯逻辑可单测。"""
    centers, assign = [], []
    for v in fps:
        best, best_d = -1, 1e9
        for ci, c in enumerate(centers):
            d = 1.0 - float(np.dot(v, c))
            if d < best_d:
                best, best_d = ci, d
        if best >= 0 and best_d < threshold:
            assign.append(best)
            # 簇心滑动更新
            centers[best] = centers[best] * 0.9 + v * 0.1
            centers[best] /= max(np.linalg.norm(centers[best]), 1e-6)
        else:
            centers.append(v.copy())
            assign.append(len(centers) - 1)
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--out", default=None, help="默认原地覆盖 labels")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.labels, encoding="utf-8")]
    fps, keep = [], []
    for r in recs:
        if (r.get("meta") or {}).get("storage") == "wds":
            v = fingerprint(r)                    # euno: WDS 首帧
        else:
            v = fingerprint(os.path.join(
                a.video_root,
                os.path.basename(r.get("video_uri", r["video_id"]))))
        if v is not None:
            fps.append(v)
            keep.append(r)
    assign = cluster(fps, a.threshold)
    for r, c in zip(keep, assign):
        r.setdefault("meta", {})["camera_id"] = f"fpcam_{c:04d}"
        r["meta"]["camera_id_source"] = "fingerprint"
    out = a.out or a.labels
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
    n_c = len(set(assign))
    sizes = sorted([assign.count(c) for c in set(assign)], reverse=True)
    print(f"{len(keep)} videos -> {n_c} pseudo cameras, sizes={sizes[:10]}"
          f"{'...' if n_c > 10 else ''} -> {out}")


if __name__ == "__main__":
    main()
