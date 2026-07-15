"""Gate B(比对侧): JAX gemma 库按"16 图拼装"构造同一输入,与 HF 基准逐位比对。

  /dev/shm/venv_jax/bin/python jax_impl/poc/02b_jax_layout.py --ref /tmp/hf_layout.json

判定(全部满足 = Gate B PASS,JAX 路线放行):
  1. 每帧 soft token 数一致(HF 侧应为 70)
  2. 视觉块数 = 帧数,块前后特殊 token 一致
  3. 文本段 input_ids 完全一致(时间戳行需在 prompt 里手工复刻)
库 API 若与假设不符,脚本打印候选属性后退出 —— 按提示改拼装段即可。
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/hf_layout.json")
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    print(f"[ref] HF 基准: total={ref['total_len']} soft={ref['soft_token_total']} "
          f"tokens/frame={ref['tokens_per_frame']}")

    import numpy as np
    from gemma import gm

    # ---- tokenizer(内省找 Gemma4 对应类)----
    tok_cands = [n for n in dir(gm.text) if "okenizer" in n]
    print(f"[introspect] gm.text tokenizer 候选: {tok_cands}")
    tok_cls = getattr(gm.text, "Gemma4Tokenizer", None) or \
        getattr(gm.text, "Gemma3Tokenizer", None)
    if tok_cls is None:
        print("[FAIL] 未找到 tokenizer 类,按上面候选修改")
        sys.exit(1)
    tok = tok_cls()
    print(f"[OK] tokenizer: {tok_cls.__name__}")

    # ---- 视觉预处理:16 帧按 16 图,384×384(与生产一致)----
    nf, sz = ref["num_frames"], ref["image_size"]
    frames = [np.full((sz, sz, 3), 128, np.uint8) for _ in range(nf)]
    vis_cands = [n for n in dir(gm) if "vision" in n.lower()] + \
        [n for n in dir(gm.nn) if "vision" in n.lower()]
    print(f"[introspect] 视觉预处理入口候选: {vis_cands}")

    # 官方多模态约定: 文本中插 <start_of_image> 占位(Gemma3 语义,待 4 验证)
    # 生产 prompt 的时间戳行需与 HF processor 生成的完全一致 —— 从 ref 反解
    special_cands = [s for s in getattr(tok, "special_tokens", [])] \
        if hasattr(tok, "special_tokens") else []
    print(f"[introspect] special tokens(前 20): {special_cands[:20]}")

    try:
        ids = tok.encode("<start_of_image>")
        print(f"[info] '<start_of_image>' -> {ids}")
    except Exception as e:  # noqa: BLE001
        print(f"[introspect] encode 探测失败: {e}")

    # ---- 逐位比对(拼装段就绪后启用)----
    # TODO(拼装): 用库的 chat 模板 + images= 生成 input_ids 与 soft token 布局,
    # 然后:
    #   assert tokens_per_frame == ref["tokens_per_frame"]
    #   assert 视觉块前后 context == ref["block_context"]
    #   assert 文本段 ids == ref 对应段
    print("\n[NEXT] 以上内省信息就绪后,把拼装段补上再跑一次得出 PASS/NO-GO。")
    print("Gate B: PENDING(拼装段待按真实 API 补全)")


if __name__ == "__main__":
    main()
