"""训练时视频增强(training_plan 6.5 节)。

红线(代码级禁止,不只是文档约定):
  ✗ 时序翻转(倒放): k↔l(远离/靠近门廊)、placing↔taking(投递/偷盗)
    的标签会直接互换 —— 本模块不提供该函数,并在 apply 处断言帧序单调
  ✗ 时序 mixup(视频拼接): 破坏单事件语义,加权标签无意义

允许:
  ✓ 随机时序裁剪(采样前的原片窗口;生产端 5~40s 帧间隔本就浮动,
    此增强属于分布内)
  ✓ 帧 dropout 10%(用前一帧顶替,模拟丢帧)
  ✓ 空间 RandomCrop —— 必须在 resize 384 之前的原图上做,
    保证拉伸畸变模式与生产一致(4.2 节)
  ✓ ColorJitter / 水平翻转 / 亮度增强(夜间样本)
"""
import random
from typing import List, Tuple
import numpy as np


# ---------------- 纯逻辑(可无依赖测试) ----------------

def temporal_crop_range(n_total: int, ratio_range: Tuple[float, float],
                        rng: random.Random) -> Tuple[int, int]:
    """返回 [start, end) 窗口。ratio 相对全片长度。"""
    lo, hi = ratio_range
    ratio = rng.uniform(lo, hi)
    length = max(1, int(round(n_total * ratio)))
    start = rng.randint(0, n_total - length)
    return start, start + length


def frame_dropout_indices(indices: List[int], p: float,
                          rng: random.Random) -> List[int]:
    """随机把某些帧索引替换为前一保留帧(首帧不 drop)。
    输出仍保持非递减 —— 时序方向不可逆的结构性保证。"""
    out = [indices[0]]
    for i in indices[1:]:
        out.append(out[-1] if rng.random() < p else i)
    assert all(b >= a for a, b in zip(out, out[1:])), "frame order must be monotonic"
    return out


def assert_monotonic(indices: List[int]):
    """任何增强后的帧索引必须非递减 —— 时序翻转的最后防线。"""
    if any(b < a for a, b in zip(indices, indices[1:])):
        raise AssertionError(
            "temporal order violated — 时序翻转被禁止(k↔l/placing↔taking 标签会互换)")


# ---------------- 组合入口 ----------------

def plan_augmented_indices(n_total: int, num_frames: int, cfg: dict,
                           rng: random.Random) -> List[int]:
    """训练时的帧索引计划: (可选)时序裁剪 → 均匀采样 → 帧 dropout。"""
    from data.sampling import uniform_indices
    start, end = 0, n_total
    if rng.random() < cfg.get("temporal_crop_prob", 0.0):
        start, end = temporal_crop_range(
            n_total, tuple(cfg.get("temporal_crop_ratio", (0.5, 0.9))), rng)
    idx = [start + i for i in uniform_indices(end - start, num_frames)]
    p = cfg.get("frame_dropout_prob", 0.0)
    if p > 0:
        idx = frame_dropout_indices(idx, p, rng)
    assert_monotonic(idx)
    return idx


def spatial_augment(frames: np.ndarray, cfg: dict,
                    rng: random.Random) -> np.ndarray:
    """空间增强,作用于原分辨率帧(resize 384 之前)。整段视频用同一
    crop/flip 参数(帧间空间一致性),ColorJitter 同参数应用。"""
    t, h, w = frames.shape[:3]

    # RandomCrop(原图上,占比 scale)
    lo, hi = cfg.get("spatial_crop_scale", (1.0, 1.0))
    scale = rng.uniform(lo, hi)
    if scale < 0.999:
        ch, cw = int(h * scale), int(w * scale)
        y0 = rng.randint(0, h - ch)
        x0 = rng.randint(0, w - cw)
        frames = frames[:, y0:y0 + ch, x0:x0 + cw]

    # 水平翻转(前后方向语义不受左右翻转影响,安全)
    if rng.random() < cfg.get("hflip_prob", 0.0):
        frames = frames[:, :, ::-1]

    # ColorJitter: 亮度/对比度整段同参
    cj = cfg.get("color_jitter", 0.0)
    if cj > 0:
        f = frames.astype(np.float32)
        f *= 1.0 + rng.uniform(-cj, cj)                       # 亮度
        mean = f.mean(axis=(1, 2, 3), keepdims=True)
        f = (f - mean) * (1.0 + rng.uniform(-cj, cj)) + mean  # 对比度
        frames = np.clip(f, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frames)
