"""v1.8 单测: 无空格目标/字符覆盖权重/aux 置信度门/eval_metrics 口径。
运行: python jax_impl/poc/test_v18.py
"""
import io, json, os, pickle, subprocess, sys, tarfile, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from PIL import Image
from jax_impl.data import SftDataset, EOT


class TriTok:                       # 3 字符/词元 → 制造跨界 token
    def encode(self, s):
        return [i for i in range(0, len(s), 3)] and \
               [hash(s[i:i+3]) % 50000 + 100 for i in range(0, len(s), 3)]
    _mem = {}
    def __init__(self):
        self._rev = {}
    def encode(self, s):
        out = []
        for i in range(0, len(s), 3):
            piece = s[i:i+3]
            tid = abs(hash(piece)) % 50000 + 200
            self._rev[tid] = piece
            out.append(tid)
        return out
    def decode(self, ids):
        return "".join(self._rev[t] for t in ids)


d = tempfile.mkdtemp()
buf = io.BytesIO(); Image.fromarray(np.zeros((8,8,3), np.uint8)).save(buf,"JPEG")
jpg = buf.getvalue()
with tarfile.open(os.path.join(d, "shard-000000.tar"), "w") as tf:
    raw = pickle.dumps({"frames": [jpg]*2})
    ti = tarfile.TarInfo("v0.pyd"); ti.size = len(raw)
    tf.addfile(ti, io.BytesIO(raw))
with open(os.path.join(d, "labels.jsonl"), "w") as f:
    f.write(json.dumps({"video_id": "v0",
        "labels": {"role_type": "B", "sub_keyscene": "i",
                   "description": "A man walks."},
        "meta": {"shard": 0, "wds_dir": d}}) + "\n")
with open(os.path.join(d, "layout.json"), "w") as f:
    json.dump({"input_ids": [5,6,7,8], "mm_token_type_ids": [0,0,0,0]}, f)

tok = TriTok()
ds = SftDataset(os.path.join(d,"labels.jsonl"), os.path.join(d,"layout.json"), tok)
ex = ds[0]
T = 4
lab = [t for t in ex["labels"][T:] if t != -100]
txt = tok.decode([t for t in lab if t != EOT])
assert txt == "B|i|A man walks.", txt          # 无空格 + 单次分词回环
w = ex["weights"][T:T+len(lab)]
# TriTok: "B|i" (0-3) 全在前缀 → 4;"|A " (3-6) 跨界(start 3<4)→ 4;
# 之后全 1;EOT=1
assert w[0] == 4.0 and w[1] == 4.0, w[:4]
assert all(x == 1.0 for x in w[2:]), w
assert w[len(lab)-1] == 1.0                     # EOT
print("OK 无空格目标 + 字符覆盖权重(含跨界 token)")

# ---- aux 置信度门 ----
attrs_low  = {"v0": {"attributes": {"clothing": "uniform"}, "confidence": 0.3}}
attrs_high = {"v0": {"attributes": {"clothing": "uniform"}, "confidence": 0.9}}
dl = SftDataset(os.path.join(d,"labels.jsonl"), os.path.join(d,"layout.json"),
                tok, attributes=attrs_low)
dh = SftDataset(os.path.join(d,"labels.jsonl"), os.path.join(d,"layout.json"),
                tok, attributes=attrs_high)
assert (dl[0]["aux_labels"] == -100).all(), dl[0]["aux_labels"]
assert (dh[0]["aux_labels"] != -100).any(), dh[0]["aux_labels"]
print("OK aux 置信度门(0.3 → 全屏蔽, 0.9 → 生效)")

# ---- eval_metrics 口径: 缺失记入安全召回分母 ----
lp, pp = os.path.join(d,"el.jsonl"), os.path.join(d,"ep.jsonl")
with open(lp,"w") as f:                          # 2 条安全关键(k),1 条缺预测
    for vid, sk in [("a","q"), ("b","q")]:
        f.write(json.dumps({"video_id": vid,
            "labels": {"role_type":"B","sub_keyscene":sk}})+"\n")
with open(pp,"w") as f:
    f.write(json.dumps({"video_id":"a","output":"B|q|x"})+"\n")
r = subprocess.run([sys.executable, "jax_impl/eval_metrics.py",
                    "--preds", pp, "--labels", lp],
                   capture_output=True, text=True,
                   cwd=os.path.dirname(os.path.dirname(
                       os.path.dirname(os.path.abspath(__file__)))))
out = r.stdout
assert "召回 = 50.00% (n=2)" in out, out         # 旧版会报 100% (n=1)
assert "已在全部指标中记错" in out
print("OK eval_metrics 缺失口径(召回 50%,旧版虚报 100%)")
print("ALL PASS")
