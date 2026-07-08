"""torch-free 核心逻辑单元测试(python3 -m pytest 或直接 python3 运行)。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.formatting import (build_target, build_cot_target,
                             char_spans_to_token_weights, char_weight_at)
from eval.format_validator import parse_output, is_legal, deployment_guard
from eval.metrics import evaluate
from annotation.consistency_filter import filter_gt, filter_double


def test_build_target_basic():
    t = build_target("B", "i", "A delivery person places a package.")
    assert t.text == "B | i | A delivery person places a package."
    # 分类段 = "B | i | " 共 8 字符,权重 4.0
    assert t.weight_spans == [(0, 8, 4.0)]
    assert char_weight_at(t, 0) == 4.0     # 'B'
    assert char_weight_at(t, 4) == 4.0     # 'i'
    assert char_weight_at(t, 8) == 1.0     # desc 第一个字符


def test_build_target_rejects_bad_letters():
    for bad in [("b", "i"), ("F", "i"), ("B", "I"), ("B", "z")]:
        try:
            build_target(bad[0], bad[1], "x")
            assert False, f"should reject {bad}"
        except ValueError:
            pass


def test_cot_target_think_masked():
    t = build_cot_target("C", "n", "A stranger takes the package.",
                         "[身份线索] 便服张望 [场景线索] 门廊包裹 [结论] 偷盗")
    assert t.text.startswith("<think>")
    close = t.text.index("</think>\n") + len("</think>\n")
    # think 段权重 0
    assert char_weight_at(t, 0) == 0.0
    assert char_weight_at(t, close - 1) == 0.0
    # 答案分类段权重 4.0
    assert t.text[close] == "C"
    assert char_weight_at(t, close) == 4.0
    # description 权重 1.0
    assert char_weight_at(t, close + 8) == 1.0


def test_token_weight_mapping():
    t = build_target("B", "i", "Hi.")
    # 模拟 tokenizer offsets: "B"|" |"|" i"|" |"|" Hi"(跨界)|special
    offsets = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 11), (0, 0)]
    w = char_spans_to_token_weights(t, offsets)
    assert w[0] == 4.0            # "B"
    assert w[2] == 4.0            # " i"
    assert w[4] == 4.0            # (7,11) 跨过 cls_end=8,含位置 7 → max=4.0
    assert w[5] == 0.0            # special token
    # 纯 desc token(不跨界)权重 1.0
    w2 = char_spans_to_token_weights(t, [(8, 11)])
    assert w2[0] == 1.0


def test_token_weight_crossing_boundary():
    t = build_target("B", "i", "Hi.")
    w = char_spans_to_token_weights(t, [(6, 9)])  # 跨过 cls_end=8
    assert w[0] == 4.0  # 覆盖到分类段 → 取最大权重


def test_parse_output_happy():
    r = parse_output("B | i | A delivery person places a package.")
    assert r.ok and r.rt == "B" and r.sk == "i"
    assert r.legal_combo and not r.case_fixed and not r.think_leak
    assert deployment_guard(r)["action"] == "accept"


def test_parse_output_case_fix():
    r = parse_output("c | C | someone loiters")   # 大小写全错
    assert r.ok and r.rt == "C" and r.sk == "c" and r.case_fixed


def test_parse_output_illegal_combo():
    r = parse_output("E | c | a kid plays")       # 无人 + 儿童玩耍
    assert r.ok and not r.legal_combo
    r2 = parse_output("A | n | family steals?")   # 家人 + 偷包裹
    assert r2.ok and not r2.legal_combo
    assert deployment_guard(r2)["action"] == "escalate_to_C_alert"


def test_parse_output_dq_illegal():
    """D|q 在 training_plan 14.2 非法组合表内;q 涉安全 → 升级为 C 告警。"""
    r = parse_output("D | q | an unidentified person holds a knife")
    assert r.ok and not r.legal_combo
    assert deployment_guard(r)["action"] == "escalate_to_C_alert"
    # 其他含 q 的组合不在表内,不受影响
    r2 = parse_output("C | q | a suspicious person holds a knife")
    assert r2.legal_combo
    assert deployment_guard(r2)["action"] == "accept"


def test_format_alignment_verify():
    """交付前 GT 整串字节核对(eval/check_format_alignment)。"""
    from eval.check_format_alignment import verify_gt_line
    assert verify_gt_line(
        "B | i | A delivery person places a package.") is None
    assert verify_gt_line("B|i|no spaces") is not None      # 分隔符约定不同
    assert verify_gt_line("b | i | lowercase rt") is not None  # GT 不该需矫正
    assert verify_gt_line("garbage") is not None            # 解析失败


def test_parse_output_think_leak():
    r = parse_output("<think>reasoning</think>\nC | s | person lingers")
    assert r.think_leak and r.ok and r.rt == "C" and r.sk == "s"


def test_parse_output_garbage():
    r = parse_output("no pipes here")
    assert not r.ok
    assert deployment_guard(r)["action"] == "reject"


def test_metrics_end_to_end():
    gts = {"v1": ("B", "i"), "v2": ("C", "n"), "v3": ("A", "c"),
           "v4": ("C", "s"), "v5": ("D", "m")}
    preds = {"v1": "B | i | ok",
             "v2": "A | h | wrong both",      # C->A, n->h(热区混淆)
             "v3": "A | c | ok",
             "v4": "d | S | case fixed ok",   # 大小写矫正后正确
             "v5": "garbage"}
    rep = evaluate(preds, gts)
    assert rep["n_evaluated"] == 5
    # RT: v1 B✓, v2 A✗(gt C), v3 A✓, v4 "d"→"D"✗(gt C), v5 格式失败✗
    assert rep["RoleType_acc"] == 0.4
    # SK: v1 i✓, v2 h✗(gt n), v3 c✓, v4 "S"→"s"✓, v5 ✗
    assert rep["SubKeyScene_acc"] == 0.6
    assert rep["format_fail_rate"] == 0.2    # v5
    assert rep["hotspot_confusions"]["sk:h<->n"]["n->h"] == 1
    assert rep["case_fixed_rate"] == 0.2     # v4


def test_consistency_filter_gt_mode():
    gem = {"v1": {"video_id": "v1", "gemini_output":
                  {"predictions": {"role_type": "B", "sub_keyscene": "i"}}},
           "v2": {"video_id": "v2", "gemini_output":
                  {"predictions": {"role_type": "C", "sub_keyscene": "n"}}},
           "v3": {"video_id": "v3", "gemini_output":
                  {"predictions": {"role_type": "A", "sub_keyscene": "h"}}}}
    gts = {"v1": {"labels": {"role_type": "B", "sub_keyscene": "i"}},
           "v2": {"labels": {"role_type": "C", "sub_keyscene": "s"}},   # sk 不同
           "v3": {"labels": {"role_type": "D", "sub_keyscene": "m"}}}   # 全错
    white, partial, discard = filter_gt(gem, gts)
    assert [d["video_id"] for d in white] == ["v1"]
    assert [d["video_id"] for d in partial] == ["v2"]
    assert [d["video_id"] for d in discard] == ["v3"]


def test_consistency_filter_double_mode():
    p1 = {"v1": {"video_id": "v1", "gemini_output":
                 {"predictions": {"role_type": "B", "sub_keyscene": "i"}}},
          "v2": {"video_id": "v2", "gemini_output":
                 {"predictions": {"role_type": "C", "sub_keyscene": "n"}}}}
    p2 = {"v1": {"video_id": "v1", "gemini_output":
                 {"predictions": {"role_type": "B", "sub_keyscene": "i"}}},
          "v2": {"video_id": "v2", "gemini_output":
                 {"predictions": {"role_type": "D", "sub_keyscene": "n"}}}}
    white, discard = filter_double(p1, p2)
    assert len(white) == 1 and white[0]["video_id"] == "v1"
    assert white[0]["pseudo_gt"] == {"role_type": "B", "sub_keyscene": "i"}
    assert len(discard) == 1




def test_pipeline_chain():
    """阶段串联静态检查: yaml init_from 链 + 检查点文件约定。"""
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p5b = yaml.safe_load(open(os.path.join(root, "configs/phase5b_aux.yaml")))
    p5d = yaml.safe_load(open(os.path.join(root, "configs/phase5d_cot.yaml")))
    p6 = yaml.safe_load(open(os.path.join(root, "configs/phase6_kto.yaml")))
    # 链条: 5_sft(_b) → 5b → 5d → kto
    assert "phase5_sft" in p5b["init_from"], p5b["init_from"]
    assert p5d["init_from"] == "outputs/phase5b_aux/final"
    assert p6["init_from"] == "outputs/phase5d_cot/final"
    # 白名单约定: 5b/5d 必须只用白名单(伪标签/推理链质量前提)
    assert p5b["use_whitelist_only"] and p5d["use_whitelist_only"]
    # train.py 必须同时保存和恢复 projector(跨 phase 断点的回归防线)
    src = open(os.path.join(root, "training/train.py"), encoding="utf-8").read()
    assert 'projector.pt' in src
    assert "restore_from" in src and "set_peft_model_state_dict" in src
    # kto.py 必须冻结 projector 且不用 load_adapter("default")
    ksrc = open(os.path.join(root, "training/kto.py"), encoding="utf-8").read()
    assert 'adapter_name="reference"' in ksrc
    assert ksrc.count('load_adapter(init, adapter_name="default")') == 0
    # export 必须支持训练后 projector 注入
    esrc = open(os.path.join(root, "export/split_deliverables.py"),
                encoding="utf-8").read()
    assert "--projector" in esrc
    # epoch 不可知 → 必须有早停 + best checkpoint + 追加训练入口
    assert "EarlyStoppingCallback" in src
    assert "load_best_model_at_end" in src
    assert '"--epochs"' in src and '"--resume"' in src
    import yaml as _y
    b = _y.safe_load(open(os.path.join(root, "configs/base.yaml")))
    assert b["train"]["early_stopping_patience"] >= 1
    assert b["train"]["load_best_at_end"] is True


def test_kto_stratified_batches():
    """KTO 分层 batch: 每 batch 固定混入 undesirable(纯逻辑)。"""
    import random
    from training.kto import plan_stratified_batches
    # 100 条,12 条 undesirable(≈错例占比),batch 8,每 batch 2 条错例
    is_d = [True] * 100
    for i in range(0, 96, 8):
        is_d[i] = False
    rng = random.Random(0)
    batches = plan_stratified_batches(is_d, batch_size=8,
                                      n_undesirable=2, rng=rng)
    assert all(len(b) == 8 for b in batches)
    for b in batches:
        n_u = sum(1 for i in b if not is_d[i])
        assert n_u == 2, f"batch 应含 2 条 undesirable,实际 {n_u}"
    # desirable 池决定 epoch 长度: 88 desirable / 6 per batch = 14 batches
    assert len(batches) == 14
    # 无 undesirable 时退化为普通分批
    plain = plan_stratified_batches([True] * 20, 8, 2, random.Random(0))
    assert all(len(b) == 8 for b in plain) and len(plain) == 2


def test_kto_ref_divergence_and_brake():
    from training.kto import ref_divergence_alert, classification_brake
    # 同起点前期 ref_gap≈0 正常;超过 warn_after 仍为 0 → 告警
    assert not ref_divergence_alert(0.0, step=50, warn_after=100)
    assert ref_divergence_alert(0.0, step=150, warn_after=100)
    assert not ref_divergence_alert(0.3, step=150, warn_after=100)
    # 刹车: 任一分类指标降 >0.5 → 返回劣化项
    base = {"RoleType_acc": 89.5, "SubKeyScene_acc": 81.0}
    assert classification_brake(base, {"RoleType_acc": 89.2,
                                       "SubKeyScene_acc": 80.8}) == []
    assert classification_brake(base, {"RoleType_acc": 88.9,
                                       "SubKeyScene_acc": 81.2}) == \
        ["RoleType_acc"]


def test_camera_fingerprint_cluster():
    """背景指纹贪心聚类(无机位字段数据的伪 camera_id)。"""
    import numpy as np
    from data.camera_fingerprint import cluster
    rng = np.random.RandomState(0)

    def unit(v):
        return v / np.linalg.norm(v)
    a, b = unit(rng.randn(64)), unit(rng.randn(64))
    fps = [unit(a + 0.02 * rng.randn(64)) for _ in range(5)] + \
          [unit(b + 0.02 * rng.randn(64)) for _ in range(4)]
    assign = cluster(fps, threshold=0.15)
    assert len(set(assign[:5])) == 1 and len(set(assign[5:])) == 1
    assert assign[0] != assign[5]        # 两个机位分开
    # UCSD 实测: 98 条 → 2 簇(70/28),与 ped1/ped2 真实机位零混淆


def test_hard_mining_replication():
    """hard mining 物理复制(XLA 安全的采样加权替代)。"""
    from training.train import apply_hard_mining
    recs = [{"video_id": "a"}, {"video_id": "b"}, {"video_id": "c"}]
    out = apply_hard_mining(recs, {"b": 3.0})
    ids = [r["video_id"] for r in out]
    assert ids.count("b") == 3 and ids.count("a") == 1 and len(out) == 5




def test_pipeline_stage_helper():
    """逐阶段助手: 状态判定 + init_from 磁盘缝合 + 依赖校验(纯逻辑)。"""
    from training.pipeline import stage_status, init_for, build_cmds

    def flags(**on):
        base = {s: {"enabled": False} for s in
                ["sft_warmup", "sft_joint", "hard_mining", "aux_heads",
                 "implicit_cot", "kto", "swa"]}
        for k, v in on.items():
            base[k] = {"enabled": v}
        return {"stages": base, "num_cores": 1,
                "start_from_checkpoint": None}

    data = {"labels_file": "L", "whitelist_file": "W",
            "attributes_file": "/no_A", "reasoning_file": "/no_C"}

    # 磁盘状态 mock: sft_joint 已完成
    done = {"outputs/phase5_sft_b/final"}
    ex = lambda p: p in done

    pcfg = flags(sft_joint=True, kto=True, swa=True)
    st = stage_status(pcfg, exists=ex)
    assert st["sft_joint"] == "done" and st["kto"] == "pending"
    assert st["aux_heads"] == "skipped"

    # kto 的 init 应缝合到已完成的 sft_joint(跳过 aux/cot)
    assert init_for("kto", pcfg, exists=ex) == "outputs/phase5_sft_b/final"
    cmds = build_cmds("kto", pcfg, data, exists=ex)
    assert len(cmds) == 3 and "--init-from outputs/phase5_sft_b/final" in cmds[2]

    # swa 在 kto 完成后应接 kto 的 checkpoints
    done.add("outputs/phase6_kto/final")
    cmds = build_cmds("swa", pcfg, data, exists=ex)
    assert "phase6_kto/checkpoint-*" in cmds[0]

    # 依赖校验: 资产文件不存在时 aux 报错
    pcfg2 = flags(sft_joint=True, aux_heads=True)
    try:
        build_cmds("aux_heads", pcfg2, data, exists=ex)
        assert False, "should require attributes file"
    except ValueError as e:
        assert "属性文件" in str(e)

    # 外部起点: 磁盘无任何产物时用 start_from_checkpoint
    pcfg3 = flags(kto=True)
    pcfg3["start_from_checkpoint"] = "outputs/external_v15/final"
    assert init_for("kto", pcfg3, exists=lambda p: False) == \
        "outputs/external_v15/final"


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
