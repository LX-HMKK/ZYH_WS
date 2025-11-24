"""基础用法示例。

功能新增：
1. 自动列出可用串口
2. 让用户输入序号或自定义端口名

运行前请确认已连接硬件；若自动扫描为空，可手动输入 (例如 COM3)。
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
import re

# 无论 cwd/调试/命令行，始终将项目根目录插入 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

from gloria_msdk import SerialManager, SerialConfig, Command  # noqa: E402


def choose_port() -> str:
	try:
		from serial.tools import list_ports  # type: ignore
	except Exception:  # noqa: BLE001
		# 未安装 pyserial 或其他错误，交互输入
		try:
			val = input("请输入串口名 (直接回车退出，示例 COM3): ").strip()
		except KeyboardInterrupt:
			print("\n用户取消输入。")
			return ""
		return val

	ports = list(list_ports.comports())
	if not ports:
		print("未发现可用串口。")
		try:
			val = input("请手动输入端口名 (直接回车退出，示例 COM3): ").strip()
		except KeyboardInterrupt:
			print("\n用户取消输入。")
			return ""
		return val
	print("发现以下串口:")
	for idx, p in enumerate(ports):
		desc = p.description or ""
		hwid = p.hwid or ""
		print(f"[{idx}] {p.device} - {desc} ({hwid})")
	while True:
		try:
			sel = input("请选择序号，或直接输入端口名 (回车退出): ").strip()
		except KeyboardInterrupt:
			print("\n用户取消选择。")
			return ""
		if sel == "":
			# 空回车直接退出
			return ""
		if sel.isdigit():
			num = int(sel)
			if 0 <= num < len(ports):
				return ports[num].device
			else:
				print(f"序号无效，请输入0到{len(ports)-1}之间的数字")
				continue  # 重新循环等待输入
		# 用户直接输入端口名
		if (re.match(r'^COM\d+$', sel) or  # Windows: COM1, COM10等
            re.match(r'^/dev/tty(USB|ACM|S)\d+$', sel)):  # Linux/macOS
			return sel
		else:
			print("端口名格式似乎不正确，请重新输入。")
			continue

def main():
	try:
		# 方法优先级: 命令行参数 > 环境变量 GLORIA_PORT > 交互选择
		if len(sys.argv) > 1:
			port = sys.argv[1].strip()
		elif os.getenv("GLORIA_PORT"):
			port = os.getenv("GLORIA_PORT").strip()
		else:
			port = choose_port()
		if not port:
			print("未提供端口，退出。")
			return
		cfg = SerialConfig(port=port, baudrate=921600)
		mgr = SerialManager(cfg)

		# 注册回调
		mgr.on_status.register(lambda st: print("[STATUS]", st))
		mgr.on_ack.register(lambda cmd: print(f"[ACK] command=0x{cmd.code:02X}"))
		mgr.on_error.register(lambda err: print("[ERROR]", err))

		print(f"打开串口 {port} ... (Ctrl+C 可安全退出)")
		mgr.open()
		try:
			while True:
				print("发送示例命令 0x01")
				mgr.send_command(Command(code=0x01, params=b""))
				print("等待状态回传 (示例运行 3 秒)... 按 Ctrl+C 可提前终止")
				import time
				t0 = time.time()
				while time.time() - t0 < 3:
					# 主动解析接收缓冲区
					if mgr._reader:
						mgr._reader.consume()
					time.sleep(0.05)
		except KeyboardInterrupt:
			print("\n检测到 Ctrl+C，中止示例...")
		finally:
			mgr.close()
			print("已关闭串口")
	except KeyboardInterrupt:
		print("\n用户取消。")


if __name__ == "__main__":  # pragma: no cover
	main()
