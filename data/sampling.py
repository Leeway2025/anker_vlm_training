"""视频采样与预处理(与生产逐字对齐,training_plan 4.1/4.2 节,均已客户确认)。

生产规则(不可改):
  ① 全片均匀采样 16 帧(非固定 FPS;帧间隔随 5~40s 时长在 0.31~2.5s 浮动)
  ② 直接 resize 到 384×384(拉伸,不保纵横比;基线即此畸变分布)

纯逻辑函数(uniform_indices)与解码分离,便于无依赖单元测试。
"""
from typing import List, Optional, Tuple
import numpy as np


def uniform_indices(n_total: int, n_sample: int) -> List[int]:
    """全片均匀取帧索引(含首尾)。n_total < n_sample 时重复末帧补齐。"""
    if n_total <= 0:
        raise ValueError("empty video")
    if n_total >= n_sample:
        return np.linspace(0, n_total - 1, n_sample).round().astype(int).tolist()
    idx = list(range(n_total))
    return idx + [n_total - 1] * (n_sample - n_total)


def decode_video(path: str, indices: Optional[List[int]] = None,
                 num_frames: int = 16) -> Tuple[np.ndarray, Tuple[int, int]]:
    """解码指定帧。返回 (frames[T,H,W,3] uint8 RGB, (orig_w, orig_h))。

    decord 优先(快),cv2 兜底。indices=None 时全片均匀 num_frames 帧。
    """
    try:
        import decord
        vr = decord.VideoReader(path, num_threads=2)
        n = len(vr)
        idx = indices if indices is not None else uniform_indices(n, num_frames)
        frames = vr.get_batch(idx).asnumpy()          # RGB
        h, w = frames.shape[1:3]
        return frames, (w, h)
    except ImportError:
        pass

    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        raise IOError(f"cannot read frame count: {path}")
    idx = indices if indices is not None else uniform_indices(n, num_frames)
    want, out = set(idx), {}
    for i in range(max(idx) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    last = None
    frames = []
    for i in idx:                     # 缺帧用最近成功帧补
        last = out.get(i, last)
        if last is None:
            raise IOError(f"no decodable frame in {path}")
        frames.append(last)
    arr = np.stack(frames)
    h, w = arr.shape[1:3]
    return arr, (w, h)


def resize_production(frames: np.ndarray, size: int = 384) -> np.ndarray:
    """生产预处理: 逐帧直接 resize(拉伸)。禁止改成 letterbox/crop——
    生产端就是拉伸,训练必须同分布。"""
    import cv2
    return np.stack([cv2.resize(f, (size, size),
                                interpolation=cv2.INTER_LINEAR)
                     for f in frames])
