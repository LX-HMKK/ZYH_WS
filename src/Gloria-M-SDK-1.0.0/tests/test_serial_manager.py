from gloria_msdk.core.serial_manager import SerialManager, SerialConfig
from gloria_msdk.models.command import Command


class DummySerial:
	def __init__(self, *args, **kwargs):
		self.opened = True
		self.buffer = bytearray()
		self.to_read = bytearray()

	# 模拟读：返回空，保持线程运行
	def read(self, n: int):  # noqa: D401
		return b""

	def write(self, data: bytes):  # noqa: D401
		self.buffer.extend(data)

	def close(self):  # noqa: D401
		self.opened = False


def test_serial_manager_open_send_close():
	cfg = SerialConfig(port="MOCK")
	mgr = SerialManager(cfg, serial_cls=DummySerial)
	mgr.open()
	assert mgr.is_open()
	mgr.send_command(Command(code=1))
	mgr.close()
	assert not mgr.is_open()
