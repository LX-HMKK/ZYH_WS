# gripper_control.py
from ast import main
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
        

        # self.MotorControl1.set_zero_position(self.Motor1) # 保存零点位置
         # 保存电机参数并使能
        self.MotorControl1.save_motor_param(self.Motor1)
        self.MotorControl1.enable(self.Motor1)
    
    def open_gripper(self):
        """
        打开夹爪 - 发送打开指令
        """
        # KP, KD, POS, V, TOR
        self.MotorControl1.controlMIT(self.Motor1, 0.5, 0.5, 3.5, 0.4, 0.3)#3.5  0.2  1
        time.sleep(0.001)
    
    def close_gripper(self):
        """
        闭合夹爪 - 发送闭合指令
        """
        # KP, KD, POS, V, TOR
        self.MotorControl1.controlMIT(self.Motor1, 0.4, 0.5, 0, 0.5, -1.3)
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




def main():
    """
    主函数 - 演示夹爪控制器的基本使用方法
    """
    try:
        # 初始化夹爪控制器，请根据实际串口设备修改端口号
        # Windows系统通常是 'COM3', 'COM4' 等，Linux/Mac系统是 '/dev/ttyUSB0', '/dev/ttyACM0' 等
        #init_gripper('COM3')
        init_gripper('/dev/ttyACM0')
        print("夹爪控制器初始化成功")
        
        # 循环执行打开和闭合操作
        for i in range(1):
            print(f"第{i+1}次循环")
            
            # 打开夹爪
            print("打开夹爪...")
            open_gripper()
            time.sleep(2)  # 等待2秒
            
            # 闭合夹爪
            print("闭合夹爪...")
            close_gripper()
            time.sleep(2)  # 等待2秒
            
        print("演示完成")
        
    except Exception as e:
        print(f"发生错误: {e}")
        
    finally:
        # 清理资源
        cleanup_gripper()
        print("资源已清理")

if __name__ == "__main__":
    main()