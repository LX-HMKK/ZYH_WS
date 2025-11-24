# gripper_control.py
import math
from DM_CAN import *
import serial
import time

class GripperController:
    def __init__(self, serial_port='/dev/ttyACM0'):
        """
        初始化夹爪控制器
        
        Args:
            serial_port (str): 串口设备路径
        """
        self.Motor1 = Motor(DM_Motor_Type.DM4310, 0x01, 0x02)
        self.serial_device = serial.Serial(serial_port, 921600, timeout=0.5)
        self.MotorControl1 = MotorControl(self.serial_device)
        self.MotorControl1.addMotor(self.Motor1)
        
        # 切换到MIT控制模式
        if self.MotorControl1.switchControlMode(self.Motor1, Control_Type.MIT):
            print("switch MIT控制模式 success")
        
        # 保存电机参数并使能
        self.MotorControl1.save_motor_param(self.Motor1)
        self.MotorControl1.enable(self.Motor1)
    
    def open_gripper(self):
        """
        打开夹爪 - 发送打开指令
        """
        # KP, KD, POS, V, TOR
        self.MotorControl1.controlMIT(self.Motor1, 0.5, 0.5, 3.5, 0.1, 1)
        time.sleep(0.001)
    
    def close_gripper(self):
        """
        闭合夹爪 - 发送闭合指令
        """
        # KP, KD, POS, V, TOR
        self.MotorControl1.controlMIT(self.Motor1, 0.4, 0.5, 0, 0.1, -1)
        time.sleep(0.001)
    
    def close_connection(self):
        """
        关闭串口连接
        """
        self.serial_device.close()

# 全局控制器实例（可选）
_gripper_controller = None

def init_gripper(serial_port='/dev/ttyACM0'):
    """
    初始化夹爪控制器
    
    Args:
        serial_port (str): 串口设备路径
    """
    global _gripper_controller
    _gripper_controller = GripperController(serial_port)

def open_gripper():
    """
    打开夹爪 - 外部调用函数
    直接发送打开控制指令
    """
    if _gripper_controller is None:
        raise RuntimeError("Gripper controller not initialized. Call init_gripper() first.")
    _gripper_controller.open_gripper()

def close_gripper():
    """
    闭合夹爪 - 外部调用函数
    直接发送闭合控制指令
    """
    if _gripper_controller is None:
        raise RuntimeError("Gripper controller not initialized. Call init_gripper() first.")
    _gripper_controller.close_gripper()

def cleanup_gripper():
    """
    清理资源，关闭串口连接
    """
    global _gripper_controller
    if _gripper_controller is not None:
        _gripper_controller.close_connection()
        _gripper_controller = None