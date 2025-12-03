"""自定义异常类型。"""

class GloriaMError(Exception):
	"""SDK 基础异常。"""


class SerialConnectionError(GloriaMError):
	"""串口连接失败或状态异常。"""


class ProtocolError(GloriaMError):
	"""协议帧格式错误。"""


class CRCMismatchError(ProtocolError):
	"""CRC 校验失败。"""


class FrameTooShortError(ProtocolError):
	"""帧长度不足。"""


class PayloadTooLargeError(ProtocolError):
	"""Payload 超过协议规定大小。"""


class AckTimeoutError(GloriaMError):
	"""等待 ACK 超时。"""


class CommandFormatError(GloriaMError):
	"""命令格式无效。"""


__all__ = [
	"GloriaMError",
	"SerialConnectionError",
	"ProtocolError",
	"CRCMismatchError",
	"FrameTooShortError",
	"PayloadTooLargeError",
	"AckTimeoutError",
	"CommandFormatError",
]
