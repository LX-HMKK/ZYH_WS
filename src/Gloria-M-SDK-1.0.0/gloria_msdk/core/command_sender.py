"""命令发送线程。"""
from __future__ import annotations

import threading
import queue
from typing import Optional

from ..models.command import Command
from ..constants import FRAME_START
from ..utils import StoppableThread, crc16_modbus


class CommandSender(StoppableThread):
	def __init__(self, ser) -> None:
		super().__init__(name="CommandSender")
		self._ser = ser
		self._q: "queue.Queue[Command]" = queue.Queue()
		self._seq = 0

	def enqueue(self, cmd: Command):
		self._seq += 1
		cmd.seq = self._seq
		self._q.put(cmd)

	def run(self):  # noqa: D401
		while not self.stopped:
			try:
				cmd = self._q.get(timeout=0.1)
			except queue.Empty:
				continue
			try:
				self._send(cmd)
			except Exception:  # noqa: BLE001
				# TODO: 记录日志或回调错误
				pass

	def _send(self, cmd: Command):
		payload = cmd.to_payload()
		if len(payload) > 255:
			raise ValueError("命令 payload 过长")
		header = bytes([FRAME_START, cmd.FRAME_TYPE, len(payload)])
		crc = crc16_modbus(header[1:] + payload)
		frame = header + payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
		self._ser.write(frame)


__all__ = ["CommandSender"]
