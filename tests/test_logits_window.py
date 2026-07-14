# -*- coding: utf-8 -*-
"""logits_window 切片对位验证: 窗口路径与全序列路径的加权 CE 逐位等价。
(trainer.compute_loss 两条路径的纯数学镜像;不依赖 TPU/模型)"""
import torch
import torch.nn.functional as F


def _loss(sl, sb, sw, V):
    ce = F.cross_entropy(sl.reshape(-1, V), sb.reshape(-1),
                         reduction="none", ignore_index=-100).view(sb.shape)
    v = (sb != -100).float()
    return (ce * sw * v).sum() / (sw * v).sum()


def test_window_equals_full():
    torch.manual_seed(0)
    B, L, V, K = 2, 64, 50, 24
    logits = torch.randn(B, L, V)
    labels = torch.full((B, L), -100)
    weights = torch.zeros(B, L)
    labels[:, 45:52] = torch.randint(0, V, (B, 7))     # 右填充布局
    weights[:, 45:52] = torch.tensor([4., 4., 4., 1., 1., 1., 1.])
    full = _loss(logits[:, :-1], labels[:, 1:], weights[:, 1:], V)
    wl = logits[:, L - K:]                              # logits_to_keep=K
    assert (labels[:, :L - K + 1] == -100).all()        # collator 断言
    win = _loss(wl[:, :-1], labels[:, L - K + 1:], weights[:, L - K + 1:], V)
    assert torch.allclose(full, win, atol=1e-6), (full.item(), win.item())
    print("window==full OK")


if __name__ == "__main__":
    test_window_equals_full()
    print("1/1 passed")
