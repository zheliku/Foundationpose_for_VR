"""通信传输层导出。

本包仅处理 socket 收发，不处理业务协议。
推荐通过本入口统一导入：

    from zmq_utils.communicate import PayloadSender, PayloadReceiver
"""

from .sender import PayloadSender
from .receiver import PayloadReceiver

__all__ = [
    "PayloadSender",
    "PayloadReceiver",
]
