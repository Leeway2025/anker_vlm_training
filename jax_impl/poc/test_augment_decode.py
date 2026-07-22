"""增强与受限解码单测(CPU)。运行: python jax_impl/poc/test_augment_decode.py"""
import io, json, os, pickle, sys, tarfile, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from PIL import Image
from jax_impl.data import SftDataset
from jax_impl.infer import constrained_pick, legalize_combo, RT_SET, SK_SET


class FakeTok:
    def encode(self, s): return [ord(c) for c in s][:20]
    def decode(self, ids): return "".join(chr(t) for t in ids)


d = tempfile.mkdtemp()
img = (np.arange(8*8*3) % 251).reshape(8, 8, 3).astype(np.uint8)  # 非对称图案
buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=100)
with tarfile.open(os.path.join(d, "shard-000000.tar"), "w") as tf:
    for i in range(12):
        raw = pickle.dumps({"frames": [buf.getvalue()] * 4})
        ti = tarfile.TarInfo(f"v{i}.pyd"); ti.size = len(raw)
        tf.addfile(ti, io.BytesIO(raw))
with open(os.path.join(d, "labels.jsonl"), "w") as f:
    for i in range(12):
        f.write(json.dumps({"video_id": f"v{i}",
            "labels": {"role_type": "B", "sub_keyscene": "i", "description": "x"},
            "meta": {"camera_id": f"c{i//2}", "shard": 0, "wds_dir": d}}) + "\n")
with open(os.path.join(d, "layout.json"), "w") as f:
    json.dump({"input_ids": [5, 6, 7, 8], "mm_token_type_ids": [0, 0, 0, 0]}, f)

# ---- 增强: train 有随机变化;帧数不变;数组连续;val 恒定 ----
ds = SftDataset(os.path.join(d, "labels.jsonl"), os.path.join(d, "layout.json"),
                FakeTok(), val_n=4, seed=0, augment=True)
ti = ds.train_idx[0]
base = ds[ti]["frames"]
changed = any(not np.array_equal(ds[ti]["frames"][0], base[0]) for _ in range(20))
assert changed, "20 次取样帧从未变化 → 增强未生效"
for _ in range(10):
    fr = ds[ti]["frames"]
    assert len(fr) == 4 and all(f.flags["C_CONTIGUOUS"] for f in fr)
vi = ds.val_idx[0]
v0 = ds[vi]["frames"]
assert all(np.array_equal(ds[vi]["frames"][0], v0[0]) for _ in range(10)), \
    "val 被增强了!"
# tokens/labels 不受增强影响
t0 = ds[ti]["tokens"].tobytes()
assert all(ds[ti]["tokens"].tobytes() == t0 for _ in range(10))
print("OK 增强: train 随机/val 恒定/连续性/标签不动")

# ---- augment=False 完全旁路 ----
ds0 = SftDataset(os.path.join(d, "labels.jsonl"), os.path.join(d, "layout.json"),
                 FakeTok(), augment=False)
b0 = ds0[0]["frames"]
assert all(np.array_equal(ds0[0]["frames"][0], b0[0]) for _ in range(5))
print("OK augment=False 零行为变化")

# ---- 受限解码纯函数 ----
V = 1000
rt_ids = np.asarray([10, 11, 12, 13, 14])   # A-E
sk_ids = np.asarray(list(range(30, 51)))    # a-u
row = np.full(V, -1e9); row[777] = 99.0     # 全词表最优是非法 token
row[12] = 5.0; row[40] = 7.0
assert constrained_pick(row, 0, rt_ids, sk_ids, 60) == 12   # RT 集内最优
assert constrained_pick(row, 1, rt_ids, sk_ids, 60) == 60   # 强制 |
assert constrained_pick(row, 2, rt_ids, sk_ids, 60) == 40   # SK 集内最优
assert constrained_pick(row, 3, rt_ids, sk_ids, 60) == 60
assert constrained_pick(row, 4, rt_ids, sk_ids, 60) == 777  # desc 自由
assert legalize_combo("A", "n") == ("C", "n")
assert legalize_combo("A", "u") == ("C", "u")
assert legalize_combo("A", "c") == ("A", "c")
assert legalize_combo("D", "n") == ("D", "n")
assert len(RT_SET) == 5 and len(SK_SET) == 21
print("OK 受限解码/组合矫正")
print("ALL PASS")
