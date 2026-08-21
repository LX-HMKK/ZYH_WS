"""串口管理：负责打开/关闭串口，协调读写线程。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any
import threading
import time

try:  # 运行环境可能暂未安装 pyserial，延迟导入容错
	import serial  # type: ignore
except Exception:  # pragma: no cover
	serial = None  # type: ignore

from ..constants import (
	DEFAULT_BAUDRATE,
	DEFAULT_TIMEOUT,
	DEFAULT_WRITE_TIMEOUT,
)
from ..exceptions import SerialConnectionError
from ..utils import CallbackRegistry
from .data_reader import DataReader
from .command_sender import CommandSender


@dataclass
class SerialConfig:
	port: str
	baudrate: int = DEFAULT_BAUDRATE
	timeout: float = DEFAULT_TIMEOUT
	write_timeout: float = DEFAULT_WRITE_TIMEOUT


class SerialManager:
	"""高层封装：统一生命周期管理。

	可注册三个回调：
	- on_status(DeviceStatus)
	- on_ack(Command)
	- on_error(Exception | str)
	"""

	def __init__(
		self,
		config: SerialConfig,
		serial_cls: Optional[type] = None,
	) -> None:
		self.config = config
		self._serial_cls = serial_cls or (serial and serial.Serial)
		if self._serial_cls is None:
			raise SerialConnectionError("未找到可用 serial.Serial 类，请安装 pyserial")

		self._ser = None
		self._lock = threading.RLock()

		# 回调注册
		self.on_status = CallbackRegistry()
		self.on_ack = CallbackRegistry()
		self.on_error = CallbackRegistry()

		# 模块
		self._reader: Optional[DataReader] = None
		self._sender: Optional[CommandSender] = None
		self._running = False

	# ---------- public API ----------
	def open(self):
		with self._lock:
			if self._running:
				return
			try:
				self._ser = self._serial_cls(
					self.config.port,
					self.config.baudrate,
					timeout=self.config.timeout,
					write_timeout=self.config.write_timeout,
				)
			except Exception as e:  # noqa: BLE001
				raise SerialConnectionError(str(e)) from e

			self._sender = CommandSender(self._ser)
			self._reader = DataReader(
				self._ser,
				status_callback=self.on_status.fire,
				ack_callback=self.on_ack.fire,
				error_callback=self.on_error.fire,
			)
			self._sender.start()
			self._reader.start()
			self._running = True

	def close(self):
		with self._lock:
			if not self._running:
				return
			if self._reader:
				self._reader.stop()
			if self._sender:
				self._sender.stop()
			# 等待线程退出
			if self._reader:
				self._reader.join(timeout=1)
			if self._sender:
				self._sender.join(timeout=1)
			try:
				if self._ser:
					self._ser.close()
			finally:
				self._running = False

	def send_command(self, command, wait_ack: bool = False, timeout: float = 1.0) -> bool:
		from ..models.command import Command  # 局部导入避免循环
		if not isinstance(command, Command):
			raise TypeError("command 类型错误")
		if not self._sender:
			raise SerialConnectionError("串口未打开")
		event = None
		if wait_ack:
			event = threading.Event()

			def _ack_cb(cmd):
				if cmd.code == command.code:
					event.set()

			handle = self.on_ack.register(_ack_cb)
		self._sender.enqueue(command)
		if wait_ack:
			ok = event.wait(timeout=timeout)
			handle.dispose()  # type: ignore[name-defined]
			return ok
		return True

	def is_open(self) -> bool:
		return self._running

	# context manager
	def __enter__(self):
		self.open()
		return self

	def __exit__(self, exc_type, exc, tb):  # noqa: D401
		self.close()
		return False


__all__ = ["SerialManager", "SerialConfig"]
