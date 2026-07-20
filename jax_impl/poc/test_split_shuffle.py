"""val 切分/hard-mining/val-无-CoT 的行为单测(纯 CPU,合成数据)。
运行: python jax_impl/poc/test_split_shuffle.py
"""
import io
import json
import os
import pickle
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from PIL import Image

from jax_impl.data import SftDataset, split_by_camera


class FakeTok:
    def encode(self, s):
        return [ord(c) % 200 + 1 for c in s][:20]


d = tempfile.mkdtemp()
buf = io.BytesIO(); Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, "JPEG")
jpg = buf.getvalue()
N = 40
with tarfile.open(os.path.join(d, "shard-000000.tar"), "w") as tf:
    for i in range(N):
        raw = pickle.dumps({"frames": [jpg] * 2})
        ti = tarfile.TarInfo(f"v{i:03d}.pyd"); ti.size = len(raw)
        tf.addfile(ti, io.BytesIO(raw))
with open(os.path.join(d, "labels.jsonl"), "w") as f:
    for i in range(N):
        f.write(json.dumps({
            "video_id": f"v{i:03d}",
            "labels": {"role_type": "B", "sub_keyscene": "i",
                       "description": "x"},
            "meta": {"camera_id": f"cam{i // 4}", "shard": 0,
                     "wds_dir": d}}) + "\n")
with open(os.path.join(d, "layout.json"), "w") as f:
    json.dump({"input_ids": [5, 6, 7, 8], "mm_token_type_ids": [0, 0, 0, 0]}, f)

# ---- split_by_camera: 整组切、无交集、seed 稳定 ----
recs = [json.loads(l) for l in open(os.path.join(d, "labels.jsonl"))]
tr, va = split_by_camera(recs, 8, seed=0)
tr2, va2 = split_by_camera(recs, 8, seed=0)
assert [r["video_id"] for r in va] == [r["video_id"] for r in va2]  # 稳定
assert len(va) == 8 and len(tr) == N - 8
cams_tr = {r["meta"]["camera_id"] for r in tr}
cams_va = {r["meta"]["camera_id"] for r in va}
assert not (cams_tr & cams_va), "camera 横跨 train/val"
print("OK split_by_camera")

# ---- 先切后复制: 副本只在 train,val 不含副本 ----
sw = {f"v{i:03d}": 3.0 for i in range(N)}            # 全员 3 份
ds = SftDataset(os.path.join(d, "labels.jsonl"), os.path.join(d, "layout.json"),
                FakeTok(), sample_weights=sw, val_n=8, seed=0)
assert len(ds.val_idx) == 8                           # val 未膨胀
assert len(ds.train_idx) == (N - 8) * 3               # train ×3
val_vids = {ds.recs[i]["video_id"] for i in ds.val_idx}
train_vids = {ds.recs[i]["video_id"] for i in ds.train_idx}
assert not (val_vids & train_vids), "泄漏: 同视频跨 train/val"
print("OK 先切后复制")

# ---- 最大余数法: w=1.3 → 总量≈Σw(round 会全截成 1)----
sw13 = {f"v{i:03d}": 1.3 for i in range(N)}
ds13 = SftDataset(os.path.join(d, "labels.jsonl"), os.path.join(d, "layout.json"),
                  FakeTok(), sample_weights=sw13, val_n=0, seed=0)
assert len(ds13.recs) in (int(N * 1.3), int(N * 1.3) + 1), len(ds13.recs)
print("OK 最大余数法", len(ds13.recs))

# ---- val 无 CoT: 100 次取样恒定;train 侧应命中过 CoT ----
reasoning = {f"v{i:03d}": {"reasoning_chain": "because reasons " * 5}
             for i in range(N)}
dsc = SftDataset(os.path.join(d, "labels.jsonl"), os.path.join(d, "layout.json"),
                 FakeTok(), reasoning=reasoning, cot_ratio=0.9,
                 val_n=8, seed=0)
vi = dsc.val_idx[0]
ref = dsc[vi]["tokens"].tobytes()
assert all(dsc[vi]["tokens"].tobytes() == ref for _ in range(100)), \
    "val 样本内容漂移"
assert not (dsc[vi]["weights"] == 0)[4:24].all() or \
    dsc[vi]["labels"][4] != -100  # val 无 think 段(紧跟模板即 cls)
hit = sum((dsc[dsc.train_idx[0]]["weights"][4] == 0.0) for _ in range(50))
assert hit > 10, "train 侧 CoT 从未注入?"
print("OK val 无 CoT(100 次取样逐位一致), train CoT 命中", hit, "/50")

# ---- 兼容性: 无 val_n/无 sw 时行为与旧版一致(infer/kto 调用形态)----
ds0 = SftDataset(os.path.join(d, "labels.jsonl"), os.path.join(d, "layout.json"),
                 FakeTok())
assert len(ds0) == N and ds0.first_val == N and ds0.val_idx == []
assert [r["video_id"] for r in ds0.recs] == [f"v{i:03d}" for i in range(N)]
print("OK 旧调用形态兼容")
print("ALL PASS")
