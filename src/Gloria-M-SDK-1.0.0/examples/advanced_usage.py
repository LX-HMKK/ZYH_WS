"""进阶用法：

1. 使用 with 上下文管理器
2. 等待 ACK
3. 自定义 mock 串口 (无硬件调试)
"""
from __future__ import annotations

import time
import sys
import os
from pathlib import Path

# 兼容直接运行路径
if __package__ is None and __name__ == "__main__":  # pragma: no cover
	parent = Path(__file__).resolve().parents[1]
	if str(parent) not in sys.path:
		sys.path.insert(0, str(parent))

from gloria_msdk import SerialManager, SerialConfig, Command, FRAME_START, FRAME_TYPE_ACK  # noqa: E402
from gloria_msdk.utils import crc16_modbus  # noqa: E402


class MockSerial:
	"""简化的 mock：write 后自动生成 ACK 帧供 read 返回。"""

	def __init__(self, *_, **__):
		self._written = []
		self._pending = bytearray()

	def write(self, data: bytes):  # noqa: D401
		self._written.append(data)
		# 解析 command code 生成 ack
		if len(data) >= 5 and data[0] == FRAME_START:  # start type len ... crc
			frame_type = data[1]
			if frame_type != 0x10:
				return
			ln = data[2]
			payload = data[3 : 3 + ln]
			# ack payload = 原 command payload
			header = bytes([FRAME_START, FRAME_TYPE_ACK, len(payload)])
			crc = crc16_modbus(header[1:] + payload)
			ack_frame = header + payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
			self._pending.extend(ack_frame)

	def read(self, n: int):  # noqa: D401
		if not self._pending:
			return b""
		out = self._pending[:n]
		del self._pending[:n]
		return bytes(out)

	def close(self):  # noqa: D401
		pass


def demo_wait_ack():
	# 允许自定义端口(默认 MOCK)，方便替换真实串口测试 ACK 行为
	if len(sys.argv) > 1:
		port = sys.argv[1].strip()
	elif os.getenv("GLORIA_PORT"):
		port = os.getenv("GLORIA_PORT").strip()
	else:
		port = "MOCK"
	cfg = SerialConfig(port=port)
	print(f"使用端口: {port} (Ctrl+C 可安全终止)")
	try:
		with SerialManager(cfg, serial_cls=MockSerial if port == "MOCK" else None) as mgr:
			mgr.on_ack.register(lambda cmd: print(f"收到 ACK: 0x{cmd.code:02X}"))
			ok = mgr.send_command(Command(code=0x05, params=b"XY"), wait_ack=True, timeout=1.0)
			print("等待 ACK 结果:", ok)
			time.sleep(0.5)
	except KeyboardInterrupt:
		print("\n用户中断，安全退出。")


if __name__ == "__main__":  # pragma: no cover
	demo_wait_ack()
