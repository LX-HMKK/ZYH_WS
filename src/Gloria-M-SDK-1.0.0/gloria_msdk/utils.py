"""通用工具函数。"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Iterable, Optional


def crc16_modbus(data: bytes) -> int:
	"""计算 MODBUS CRC16 (多项式 0xA001)。返回 0-0xFFFF。"""
	crc = 0xFFFF
	for b in data:
		crc ^= b
		for _ in range(8):
			if crc & 1:
				crc = (crc >> 1) ^ 0xA001
			else:
				crc >>= 1
	return crc & 0xFFFF


def now_ts() -> float:
	return time.time()


class StoppableThread(threading.Thread):
	"""带 stop() 标记的线程基类。"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._stop_event = threading.Event()
		self.daemon = True

	def stop(self):  # type: ignore[override]
		self._stop_event.set()

	@property
	def stopped(self) -> bool:
		return self._stop_event.is_set()


def chunk_iter(data: bytes, size: int) -> Iterable[bytes]:
	for i in range(0, len(data), size):
		yield data[i : i + size]


@dataclass
class CallbackHandle:
	remove: Callable[[], None]

	def dispose(self):
		self.remove()


class CallbackRegistry:
	"""线程安全的回调注册。"""

	def __init__(self):
		self._lock = threading.RLock()
		self._callbacks: list[Callable] = []

	def register(self, cb: Callable) -> CallbackHandle:
		with self._lock:
			self._callbacks.append(cb)

		def _remove():
			with self._lock:
				try:
					self._callbacks.remove(cb)
				except ValueError:
					pass

		return CallbackHandle(remove=_remove)

	def fire(self, *args, **kwargs):
		with self._lock:
			callbacks = list(self._callbacks)
		for cb in callbacks:
			try:
				cb(*args, **kwargs)
			except Exception:  # noqa: BLE001
				# 生产环境可添加日志
				pass


__all__ = [
	"crc16_modbus",
	"now_ts",
	"StoppableThread",
	"chunk_iter",
	"CallbackRegistry",
	"CallbackHandle",
]
