from gloria_msdk.core.data_reader import DataReader
from gloria_msdk.constants import (
	FRAME_START,
	FRAME_TYPE_STATUS,
)
from gloria_msdk.utils import crc16_modbus


class DummySerial:
	def __init__(self, frames: list[bytes]):
		self._data = b"".join(frames)

	def read(self, n: int):  # noqa: D401
		if not self._data:
			return b""
		chunk = self._data[: n]
		self._data = self._data[n:]
		return chunk


def make_status_frame(payload: bytes) -> bytes:
	header = bytes([FRAME_START, FRAME_TYPE_STATUS, len(payload)])
	crc = crc16_modbus(header[1:] + payload)
	return header + payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def test_data_reader_status_parse():
	# 构造一个 position=10, force=20, error=0 的状态帧
	import struct

	payload = struct.pack("<HHB", 10, 20, 0)
	frame = make_status_frame(payload)
	ser = DummySerial([frame])
	received = []

	reader = DataReader(
		ser,
		status_callback=lambda s: received.append(s),
		ack_callback=lambda c: None,
		error_callback=lambda e: None,
	)
	reader.start()
	import time

	time.sleep(0.2)
	reader.stop()
	reader.join(timeout=1)
	assert received, "应解析到状态"
	st = received[0]
	assert st.position == 10
	assert st.force == 20
	assert st.error == 0
