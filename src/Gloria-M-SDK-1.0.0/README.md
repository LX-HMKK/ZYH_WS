# Gloria-M-SDK

Gloria-M 夹爪 Python SDK —— 串口双线程(读/写) + 自定义轻量协议。

## 特性

- 串口读写分离线程，线程安全发送队列
- 自定义帧 (Start | Type | Length | Payload | CRC16) 易调试
- 回调式事件：状态 / ACK / 错误
- 简单的命令与状态数据模型
- 可注入 mock 串口，方便无硬件测试

## 安装

```bash
pip install gloria-msdk
```

## 快速示例

```python
from gloria_msdk import SerialManager, SerialConfig, Command

cfg = SerialConfig(port="COM3")
with SerialManager(cfg) as mgr:
	mgr.on_status.register(lambda st: print("status", st))
	mgr.send_command(Command(code=0x01))
```

更多内容见 `docs/` 与 `examples/`。

## 使用步骤 (Step by Step)

### 1. 克隆项目
```powershell
git clone https://github.com/Synria-Robotics/Gloria-M-SDK.git
cd Gloria-M-SDK
```

### 2. 创建并激活虚拟环境 (可选但推荐)
使用 venv：
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
```
或使用 conda：
```powershell
conda create -n gloria python=3.11 -y
conda activate gloria
```

### 3. 安装依赖 / 开发模式
```powershell
pip install -r requirements.txt
pip install -e .
```
快速验证：
```powershell
python -c "import gloria_msdk; print('SDK OK')"
```

### 4. 运行基础示例 (三种端口指定方式)
交互模式 (自动列出串口)：
```powershell
python examples/basic_usage.py
```
命令行直接指定端口：
```powershell
python examples/basic_usage.py COM3
```
使用环境变量：
```powershell
$env:GLORIA_PORT="COM3"; python examples/basic_usage.py
```
（若已 `pip install -e .`，无需关心示例内的 `sys.path` 逻辑；那部分仅为直接从源码目录执行提供兼容。）

### 5. 运行进阶示例 (Mock ACK)
无需真实硬件：
```powershell
python examples/advanced_usage.py
```

### 6. 在你的代码中集成
```python
from gloria_msdk import SerialManager, SerialConfig, Command

cfg = SerialConfig(port="COM3")
with SerialManager(cfg) as mgr:
	mgr.on_status.register(lambda st: print("status", st))
	mgr.send_command(Command(code=0x01))
```

### 7. 运行测试
```powershell
pip install pytest
pytest -q
```

### 8. 打包 (可选发布)
```powershell
python setup.py sdist bdist_wheel
```
产物位于 `dist/`。

### 9. 快速列出本机串口
```powershell
python - <<'PY'
from serial.tools import list_ports
print([p.device for p in list_ports.comports()])
PY
```

### 10. 常见问题 (FAQ)
| 问题 | 说明 / 处理 |
|------|--------------|
| 未列出串口 | 检查设备驱动、数据线；手动输入端口名；设备管理器确认端口号 |
| 读不到状态 | 确认波特率一致、设备已开始上报；可用串口调试助手对比 |
| CRC mismatch | 说明帧损坏或协议不匹配；检查线缆/波特率/端口是否被其它程序占用 |
| Windows 权限 | 以管理员运行或释放被占用串口 (关闭其它串口工具) |
| 等待 ACK 超时 | 设备未返回 ACK；确认设备固件是否支持命令或延长超时 |

### 11. 日志/调试建议
当前示例未启用 logging，可在 `CommandSender._send` 与 `DataReader._handle_frame` 中加入 `print` 或集成 `logging` 模块进行诊断。

### 12. 自定义 / 扩展
- 新增命令：在上层直接构造不同 `Command(code=...)`。
- 新增状态字段：修改 `DeviceStatus` 的打包/解包格式，并同步固件。
- 序列号 ACK 匹配：可在协议 payload 中加入 seq 字段并在 `send_command` 按 seq 对应。

## 测试

```bash
pytest -q
```

## 许可

MIT (若需变更可更新此节)。

