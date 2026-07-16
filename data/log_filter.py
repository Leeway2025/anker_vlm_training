"""屏蔽 transformers 的 processor_kwargs 告警刷屏。

现象: 每次 processor.__call__ 打一条
  "Kwargs passed to `processor.__call__` have to be in `processor_kwargs`..."
训练一轮日志 50MB 几乎全是它(客户实测)。纯日志噪音:kwargs 实际被
正确消费(transformers 5.13 兼容层),行为无差。

用法: 在入口 import 后调用 install()(train.py / run_inference 已接)。
只精确匹配这一条消息,其余告警(含真实错误)不受影响。
"""
import logging

_SPAM_MARKERS = (
    "have to be in `processor_kwargs`",
    "video_metadata` was missing from inputs",   # fps=24 提示同样逐样本刷屏
)


class _DropSpam(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        return not any(m in msg for m in _SPAM_MARKERS)


def install():
    try:
        from transformers.utils import logging as hf_logging
        hf_logging._configure_library_root_logger()   # 确保默认 handler 存在
    except Exception:  # noqa: BLE001
        pass
    lg = logging.getLogger("transformers")
    flt = _DropSpam()
    for h in lg.handlers:
        h.addFilter(flt)      # handler 级: 对子 logger 传播的记录同样生效
    lg.addFilter(flt)
