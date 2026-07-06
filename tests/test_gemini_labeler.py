"""gemini_labeler 的 mock 测试(不需要真实 API/SDK)。

覆盖标注管线最易翻车的三块:
  1. label_one 的重试逻辑(格式非法重试 / API 异常退避 / 耗尽)
  2. main 的断点续跑(已完成样本跳过,不重复计费)
  3. 错误分流(.errors 文件,坏样本不污染主输出)
"""
import json
import os
import sys
import tempfile
import types as pytypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- mock google.genai(在 import gemini_labeler 之前注入)----
_fake_google = pytypes.ModuleType("google")
_fake_genai = pytypes.ModuleType("google.genai")
_fake_types = pytypes.ModuleType("google.genai.types")


class _FakePart:
    @staticmethod
    def from_bytes(data=None, mime_type=None):
        return {"bytes": len(data or b""), "mime": mime_type}


class _FakeGenConfig:
    def __init__(self, **kw):
        self.kw = kw


_fake_types.Part = _FakePart
_fake_types.GenerateContentConfig = _FakeGenConfig


class _FakeModels:
    def __init__(self, script):
        self.script = list(script)   # 每次调用弹出一个行为
        self.calls = 0

    def generate_content(self, model=None, contents=None, config=None):
        self.calls += 1
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        resp = pytypes.SimpleNamespace()
        resp.text = json.dumps(action)
        return resp


class _FakeClient:
    def __init__(self, script=()):
        self.models = _FakeModels(script)


_fake_genai.Client = _FakeClient
_fake_google.genai = _fake_genai
sys.modules["google"] = _fake_google
sys.modules["google.genai"] = _fake_genai
sys.modules["google.genai.types"] = _fake_types

import time                                             # noqa: E402
time.sleep = lambda *_: None                            # 重试不真等

from annotation.gemini_labeler import (label_one,       # noqa: E402
                                        build_prompt, validate_record)

GOOD = {
    "attributes": {"clothing": "casual", "held_items": "package",
                   "posture": "walking", "action": "taking_item",
                   "view_type": "doorbell", "sub_role_type": "Passerby",
                   "dwell": "brief_stop"},
    "reasoning_chain": "[身份线索] x [场景线索] y [结论] z",
    "predictions": {"role_type": "C", "sub_keyscene": "n",
                    "description": ("A person in casual clothes picks up a "
                                    "package from the porch and quickly "
                                    "walks away from the house today.")},
}
BAD = {"attributes": {}, "predictions": {"role_type": "x",
                                         "sub_keyscene": "9",
                                         "description": "short"}}


def _tmp_video():
    f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    f.write(b"\x00" * 128)
    f.close()
    return f.name


def test_label_one_happy():
    client = _FakeClient([GOOD])
    d, errs = label_one(client, "m", _tmp_video(), build_prompt(), 0.1)
    assert errs is None and d["predictions"]["role_type"] == "C"
    assert client.models.calls == 1


def test_label_one_retry_on_bad_format_then_success():
    client = _FakeClient([BAD, GOOD])            # 第 1 次格式烂 → 重试成功
    d, errs = label_one(client, "m", _tmp_video(), build_prompt(), 0.1)
    assert errs is None and client.models.calls == 2


def test_label_one_api_error_backoff_then_success():
    client = _FakeClient([RuntimeError("503"), GOOD])
    d, errs = label_one(client, "m", _tmp_video(), build_prompt(), 0.1)
    assert errs is None and client.models.calls == 2


def test_label_one_exhausted_returns_errors():
    client = _FakeClient([BAD, BAD, BAD])        # 3 次全格式烂
    d, errs = label_one(client, "m", _tmp_video(), build_prompt(), 0.1,
                        max_retries=3)
    assert errs and len(errs) >= 3               # 返回错误清单,不返回 None
    client2 = _FakeClient([RuntimeError("x")] * 3)
    d2, errs2 = label_one(client2, "m", _tmp_video(), build_prompt(), 0.1,
                          max_retries=3)
    assert d2 is None and errs2                  # 纯 API 失败 → None + errors


def test_main_resume_and_error_split():
    """端到端(mock): 断点续跑跳过已完成;坏样本进 .errors。"""
    import annotation.gemini_labeler as gl
    tmpdir = tempfile.mkdtemp()
    # 3 条样本;video 文件都指向同一个临时 mp4
    vp = _tmp_video()
    labels = os.path.join(tmpdir, "labels.jsonl")
    with open(labels, "w") as f:
        for vid in ["v1", "v2", "v3"]:
            f.write(json.dumps({"video_id": vid,
                                "video_uri": os.path.basename(vp)}) + "\n")
    out = os.path.join(tmpdir, "out.jsonl")
    # 预置 v1 已完成(断点续跑应跳过 → API 只被调 2 次样本份)
    with open(out, "w") as f:
        f.write(json.dumps({"video_id": "v1", "gemini_output": GOOD}) + "\n")

    # v2 成功;v3 三次全烂 → errors
    script = [GOOD] + [BAD, BAD, BAD]
    fake_client = _FakeClient(script)

    old_argv = sys.argv
    sys.argv = ["x", "--labels", labels, "--video-root",
                os.path.dirname(vp), "--out", out]
    try:
        # main 里 genai.Client() 会构造 fake(已 mock 模块),但脚本行为
        # 由我们的 fake_client 决定 → 直接替换 Client 工厂
        _fake_genai.Client = lambda *a, **k: fake_client
        gl.main()
    finally:
        sys.argv = old_argv

    outs = [json.loads(l) for l in open(out)]
    assert {o["video_id"] for o in outs} == {"v1", "v2"}   # v1 续跑保留,v2 新增
    errs = [json.loads(l) for l in open(out + ".errors")]
    assert [e["video_id"] for e in errs] == ["v3"]
    assert fake_client.models.calls == 1 + 3     # v2 一次 + v3 三次重试


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
