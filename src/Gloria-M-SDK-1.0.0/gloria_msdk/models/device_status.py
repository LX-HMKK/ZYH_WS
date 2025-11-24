"""设备状态模型。"""
from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import ClassVar

from ..utils import now_ts
from ..constants import FRAME_TYPE_STATUS


@dataclass(slots=True)
class DeviceStatus:
	"""设备状态。

	这里假设 payload 格式：
	struct StatusPayload {
		uint16 position;   // 0-1000
		uint16 force;      // 0-1000
		uint8  error;      // 0=OK
	} (共 5 字节)
	"""

	position: int
	force: int
	error: int
	timestamp: float

	FRAME_TYPE: ClassVar[int] = FRAME_TYPE_STATUS

	@classmethod
	def from_payload(cls, payload: bytes) -> "DeviceStatus":
		if len(payload) < 5:
			raise ValueError("状态 payload 长度不足，期望 5 字节")
		position, force, error = struct.unpack_from("<HHB", payload, 0)
		return cls(position=position, force=force, error=error, timestamp=now_ts())

	def to_payload(self) -> bytes:
		return struct.pack("<HHB", self.position, self.force, self.error)

	def ok(self) -> bool:
		return self.error == 0

	def __repr__(self) -> str:  # pragma: no cover
		return (
			f"DeviceStatus(pos={self.position}, force={self.force}, error={self.error}, "
			f"ts={self.timestamp:.3f})"
		)


__all__ = ["DeviceStatus"]
