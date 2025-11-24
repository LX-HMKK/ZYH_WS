"""SDK 常量定义。

设计一个简单的帧格式 (自定义协议)：

| Start(1) | Type(1) | Length(1) | Payload(N) | CRC16(2, little-endian) |

Start: 固定 0x40 (字符 '@')
Type:  0x01=状态帧, 0x02=ACK, 0x10=命令帧, 0x7F=错误
Length: 仅 Payload 长度 (0-255)
CRC16: 对 Type + Length + Payload 做 MODBUS 多项式 (0xA001) 计算。

该设计简单、易调试、可扩展 (Type 扩展)。
"""

FRAME_START = 0x40  # '@'

# 帧类型
FRAME_TYPE_STATUS = 0x01
FRAME_TYPE_ACK = 0x02
FRAME_TYPE_COMMAND = 0x10
FRAME_TYPE_ERROR = 0x7F

# 默认串口参数
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 0.05  # 读超时 (串口 read timeout)
DEFAULT_WRITE_TIMEOUT = 0.2

# 发送命令等待 ACK 超时
ACK_WAIT_TIMEOUT = 1.0

# 解析相关
MAX_FRAME_PAYLOAD = 255
MIN_FRAME_SIZE = 1 + 1 + 1 + 2  # Start + Type + Len + CRC (无 payload 情况)

# 测试/模拟使用
MOCK_PORT_PREFIX = "MOCK://"

__all__ = [
	"FRAME_START",
	"FRAME_TYPE_STATUS",
	"FRAME_TYPE_ACK",
	"FRAME_TYPE_COMMAND",
	"FRAME_TYPE_ERROR",
	"DEFAULT_BAUDRATE",
	"DEFAULT_TIMEOUT",
	"DEFAULT_WRITE_TIMEOUT",
	"ACK_WAIT_TIMEOUT",
	"MAX_FRAME_PAYLOAD",
	"MIN_FRAME_SIZE",
	"MOCK_PORT_PREFIX",
]
