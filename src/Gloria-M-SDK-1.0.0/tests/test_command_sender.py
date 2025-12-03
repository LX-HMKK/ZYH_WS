import types
from gloria_msdk.core.command_sender import CommandSender
from gloria_msdk.models.command import Command
from gloria_msdk.constants import FRAME_START, FRAME_TYPE_COMMAND
from gloria_msdk.utils import crc16_modbus


class DummySerial:
	def __init__(self):
		self.written = bytearray()

	def write(self, data: bytes):  # noqa: D401
		self.written.extend(data)


def test_command_sender_basic():
	ser = DummySerial()
	sender = CommandSender(ser)
	sender.start()
	cmd = Command(code=0x05, params=b"ABC")
	sender.enqueue(cmd)
	# 等待线程发送
	import time

	time.sleep(0.2)
	sender.stop()
	sender.join(timeout=1)
	assert ser.written, "应已写入数据"
	# 简单校验帧结构
	frame = bytes(ser.written)
	assert frame[0] == FRAME_START
	assert frame[1] == FRAME_TYPE_COMMAND
	ln = frame[2]
	payload = frame[3 : 3 + ln]
	crc_recv = frame[-2] | (frame[-1] << 8)
	crc_calc = crc16_modbus(frame[1 : 3 + ln])
	assert crc_recv == crc_calc
	assert payload.startswith(b"\x05ABC")
