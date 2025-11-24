"""命令模型。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..constants import FRAME_TYPE_COMMAND


@dataclass(slots=True)
class Command:
	"""表示一个待发送命令。

	code: 命令码 (1 字节)
	params: 可选参数 (0-255 字节)
	seq: 序列号 (非协议字段, 仅本地跟踪)
	"""

	code: int
	params: bytes = b""
	seq: int | None = None

	FRAME_TYPE: ClassVar[int] = FRAME_TYPE_COMMAND

	def to_payload(self) -> bytes:
		if not (0 <= self.code <= 0xFF):  # noqa: PLR2004
			raise ValueError("command code 必须在 0-255")
		return bytes([self.code]) + self.params

	@classmethod
	def from_payload(cls, payload: bytes) -> "Command":
		if not payload:
			raise ValueError("payload 为空，无法解析命令")
		return cls(code=payload[0], params=payload[1:])

	def __repr__(self) -> str:  # pragma: no cover - 简单 repr
		return f"Command(code=0x{self.code:02X}, params={self.params!r}, seq={self.seq})"


__all__ = ["Command"]
