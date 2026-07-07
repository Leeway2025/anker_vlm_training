"""KTO 数学对拍测试(需 torch,CPU 即可;无 torch 时跳过并 exit 0)。

对拍对象 = TRL KTOTrainer.kto_loss 的公式(cat 两段版),逐数值核对
本仓库 torch.where 向量化实现与其等价。上 TPU 前必须全绿
(WORK_STATUS 风险清单: 自实现损失需与参考实现数值对齐)。

    python3 tests/test_kto_math.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("SKIP: torch not installed (本套件在有 torch 的环境跑,CPU 即可)")
    sys.exit(0)

from training.kto import (kto_loss, sum_logprob, roll_completions)  # noqa: E402


def trl_style_kto_loss(pol_c, pol_r, ref_c, ref_r, pol_kl, ref_kl,
                       beta, w_d, w_u):
    """TRL KTOTrainer.kto_loss 原始形态(chosen/rejected 分段 + cat)。"""
    kl = (pol_kl - ref_kl).mean().detach().clamp(min=0)
    chosen_lr = pol_c - ref_c
    chosen_losses = 1 - torch.sigmoid(beta * (chosen_lr - kl))
    rejected_lr = pol_r - ref_r
    rejected_losses = 1 - torch.sigmoid(beta * (kl - rejected_lr))
    losses = torch.cat((w_d * chosen_losses, w_u * rejected_losses), 0)
    return losses.mean(), kl


def test_kto_loss_matches_trl_formula():
    g = torch.Generator().manual_seed(0)
    for trial in range(20):
        n_c, n_r = int(torch.randint(1, 7, (1,), generator=g)), \
            int(torch.randint(1, 7, (1,), generator=g))
        pol_c = torch.randn(n_c, generator=g) * 5
        ref_c = torch.randn(n_c, generator=g) * 5
        pol_r = torch.randn(n_r, generator=g) * 5
        ref_r = torch.randn(n_r, generator=g) * 5
        pol_kl = torch.randn(n_c + n_r, generator=g)
        ref_kl = torch.randn(n_c + n_r, generator=g)
        beta, w_d, w_u = 0.1, 1.0, 1.5

        expect, expect_kl = trl_style_kto_loss(
            pol_c, pol_r, ref_c, ref_r, pol_kl, ref_kl, beta, w_d, w_u)

        # 本仓库实现: 打乱混合顺序 + is_desirable 掩码
        perm = torch.randperm(n_c + n_r, generator=g)
        logp_policy = torch.cat([pol_c, pol_r])[perm]
        logp_ref = torch.cat([ref_c, ref_r])[perm]
        is_d = torch.cat([torch.ones(n_c), torch.zeros(n_r)])[perm]
        got, logs = kto_loss(logp_policy, logp_ref, pol_kl, ref_kl,
                             is_d, beta, w_d, w_u)

        assert torch.allclose(got, expect, atol=1e-6), \
            f"trial {trial}: {got} != {expect}"
        assert torch.allclose(logs["kl"], expect_kl, atol=1e-6)


def test_kto_loss_kl_clamp_and_reduce_fn():
    # kl 为负时 clamp 到 0
    logp = torch.tensor([1.0, -1.0])
    ref = torch.tensor([0.5, -0.5])
    kl_p = torch.tensor([-3.0, -3.0])
    kl_r = torch.tensor([0.0, 0.0])
    _, logs = kto_loss(logp, ref, kl_p, kl_r,
                       torch.tensor([1.0, 0.0]), 0.1, 1.0, 1.5)
    assert logs["kl"].item() == 0.0
    # reduce_fn 在 clamp 之前生效(模拟跨核平均把负 KL 拉正)
    _, logs2 = kto_loss(logp, ref, kl_p, kl_r,
                        torch.tensor([1.0, 0.0]), 0.1, 1.0, 1.5,
                        kl_reduce_fn=lambda t: t + 5.0)
    assert abs(logs2["kl"].item() - 2.0) < 1e-6


def test_kto_loss_gradient_direction():
    """desirable 的梯度应推高 logp_policy,undesirable 推低。"""
    logp = torch.zeros(2, requires_grad=True)
    ref = torch.zeros(2)
    kl_p, kl_r = torch.zeros(2), torch.zeros(2)
    loss, _ = kto_loss(logp, ref, kl_p, kl_r,
                       torch.tensor([1.0, 0.0]), 0.1, 1.0, 1.5)
    loss.backward()
    assert logp.grad[0] < 0    # 梯度下降 → logp[0](desirable)上升
    assert logp.grad[1] > 0    # logp[1](undesirable)下降


def test_sum_logprob_matches_naive():
    g = torch.Generator().manual_seed(1)
    B, L, V = 3, 17, 41
    logits = torch.randn(B, L, V, generator=g)
    labels = torch.randint(0, V, (B, L), generator=g)
    labels[:, :5] = -100            # prompt 段
    labels[0, -3:] = -100           # 不齐长 completion

    naive = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = labels[:, 1:]
    tok = naive.gather(2, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    expect = (tok * (tgt != -100)).sum(dim=1)

    for chunk in (4, 7, 256):       # 各种分块均须一致
        got = sum_logprob(logits, labels, chunk_size=chunk)
        assert torch.allclose(got, expect, atol=1e-5), f"chunk={chunk}"


def test_sum_logprob_window():
    """window 路径: 数值与全长一致;末端越界须 clamp(TPU 真机踩过的 bug:
    window=(1800, 2056) 而 L=2048 时 logits/labels 切片长度差 1)。"""
    g = torch.Generator().manual_seed(2)
    B, L, V = 4, 64, 128
    logits = torch.randn(B, L, V, generator=g)
    labels = torch.full((B, L), -100)
    labels[:, 40:60] = torch.randint(0, V, (B, 20), generator=g)

    full = sum_logprob(logits, labels, chunk_size=16)
    win = sum_logprob(logits, labels, chunk_size=16, window=(40, 60))
    assert torch.allclose(full, win, atol=1e-5)
    # 越界窗口(60+16 > 完成段;end=100 > L=64)也必须一致
    win2 = sum_logprob(logits, labels, chunk_size=16, window=(40, 100))
    assert torch.allclose(full, win2, atol=1e-5)
    # 窗口起点为 0 时 clamp 到 1(位置 0 无前驱 logit)
    win3 = sum_logprob(logits, labels, chunk_size=16, window=(0, 64))
    assert torch.allclose(full, win3, atol=1e-5)


def test_roll_completions_mismatch():
    ids = torch.arange(6).view(3, 2)
    am = torch.ones(3, 2)
    labels = torch.arange(6).view(3, 2) + 100
    r_ids, r_am, r_labels = roll_completions(ids, am, labels)
    # 样本 i 拿到样本 i-1 的 completion(视频张量在调用方保持不动)
    assert (r_ids[0] == ids[2]).all() and (r_ids[1] == ids[0]).all()
    assert (r_labels[0] == labels[2]).all()
    assert r_am.shape == am.shape


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
