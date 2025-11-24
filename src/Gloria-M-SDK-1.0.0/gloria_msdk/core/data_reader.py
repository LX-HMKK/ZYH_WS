"""读取串口数据的线程，解析协议帧。"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from ..constants import (
	FRAME_START,
	FRAME_TYPE_STATUS,
	FRAME_TYPE_ACK,
	FRAME_TYPE_ERROR,
	MIN_FRAME_SIZE,
)
from ..utils import StoppableThread, crc16_modbus
from ..models.device_status import DeviceStatus
from ..models.command import Command
from ..exceptions import (
	CRCMismatchError,
	FrameTooShortError,
	ProtocolError,
)



class DataReader(StoppableThread):
	def __init__(
		self,
		ser,
		status_callback: Callable[[DeviceStatus], None],
		ack_callback: Callable[[Command], None],
		error_callback: Callable[[Exception | str], None],
	) -> None:
		super().__init__(name="DataReader")
		self._ser = ser
		self._buf = bytearray()
		self._status_cb = status_callback
		self._ack_cb = ack_callback
		self._err_cb = error_callback
		self._lock = threading.Lock()

	def run(self):  # 只负责读取数据到缓冲区
		while not self.stopped:
			try:
				chunk = self._ser.read(64)
			except Exception as e:
				self._err_cb(e)
				break
			if not chunk:
				continue
			with self._lock:
				self._buf.extend(chunk)

	def consume(self):
		"""主动调用，解析缓冲区帧。线程安全。"""
		with self._lock:
			try:
				self._consume()
			except Exception as e:
				self._err_cb(e)

	# 解析缓冲区中的帧
	def _consume(self):
		while True:
			# 寻找起始符
			start_index = self._buf.find(bytes([FRAME_START]))
			if start_index < 0:
				self._buf.clear()
				return
			if start_index > 0:
				del self._buf[:start_index]
			if len(self._buf) < MIN_FRAME_SIZE:
				return  # 等待更多数据
			# 至少有 Start + Type + Len
			if len(self._buf) < 3:
				return
			frame_type = self._buf[1]
			payload_len = self._buf[2]
			total_len = 1 + 1 + 1 + payload_len + 2
			if len(self._buf) < total_len:
				return
			frame = bytes(self._buf[:total_len])
			del self._buf[:total_len]
			self._handle_frame(frame_type, frame)

	def _handle_frame(self, frame_type: int, frame: bytes):
		if len(frame) < MIN_FRAME_SIZE:
			raise FrameTooShortError()
		# frame: start type len payload crc(lo hi)
		payload_len = frame[2]
		payload = frame[3 : 3 + payload_len]
		crc_recv = frame[-2] | (frame[-1] << 8)
		crc_calc = crc16_modbus(frame[1 : 3 + payload_len])
		if crc_recv != crc_calc:
			raise CRCMismatchError(
				f"CRC mismatch recv=0x{crc_recv:04X} calc=0x{crc_calc:04X}"
			)
		if frame_type == FRAME_TYPE_STATUS:
			status = DeviceStatus.from_payload(payload)
			self._status_cb(status)
		elif frame_type == FRAME_TYPE_ACK:
			cmd = Command.from_payload(payload)
			self._ack_cb(cmd)
		elif frame_type == FRAME_TYPE_ERROR:
			self._err_cb(f"设备错误帧: {payload.hex()}")
		else:  # 未知类型
			raise ProtocolError(f"未知帧类型: 0x{frame_type:02X}")


__all__ = ["DataReader"]
