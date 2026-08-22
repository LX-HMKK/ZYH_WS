# gripper_control.py
import os
from arm_control.gripper_can import *
import serial
import time

class GripperController:
    def __init__(self, serial_port=None):
        """
        初始化夹爪控制器

        Args:
            serial_port (str): 串口设备路径
        """
        if serial_port is None:
            serial_port = os.environ.get('GRIPPER_PORT', '/dev/ttyACM0')
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

