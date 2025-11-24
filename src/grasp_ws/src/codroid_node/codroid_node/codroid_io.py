import rclpy

import sys
import os
sys.path.append("/home/zyh/zyh_demo1/Gloria-M-SDK-1.0.0/motor")

from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
import socket
import json
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from codroid_msgs.msg import RobotInfo

from DM_Motor_Test import *

class CodroidIO(Node):
    def __init__(self):
        super().__init__('CodroidIO')
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('192.168.101.100', 9005)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind(('192.168.101.99', 9006))
        self.server_socket.setblocking(False)

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.subscription = self.create_subscription(JointTrajectory, 'RobotMove', self.listener_callback, qos_profile)

        self.publisher = self.create_publisher(RobotInfo, 'RobotInfo', 10)
        
        # 订阅夹爪控制话题
        self.gripper_subscription = self.create_subscription(
            String,
            'GripperControl',
            self.gripper_callback,
            10)

        self.timer = self.create_timer(0.04, self.timer_callback)
        
        # 初始化夹爪控制器
        init_gripper('/dev/ttyACM0')
        
        # 存储最近一次的目标位置
        self.latest_target_z = None
        
        # 存储上一次执行夹爪动作的z轴坐标，避免重复执行
        self.last_gripper_z = None

    def timer_callback(self):
        try:
            # print('ok')
            data, _ = self.server_socket.recvfrom(1024)  # 接收服务器响应
            json_data = json.loads(data.decode('utf-8'))
            # print(json_data)
            robot_info = RobotInfo()
            robot_info.joint_positions = json_data["joint_positions"]
            robot_info.end_positions = json_data["end_positions"]
            robot_info.state = json_data["state"]
            robot_info.fault_flag = json_data["fault_flag"]
            self.publisher.publish(robot_info)
            # print(robot_info)
            
            # 检查是否需要触发夹爪动作
            self.check_and_control_gripper(robot_info)
        except:
            return

    def listener_callback(self, msg):
        json_str = self.to_json(msg)
        print(json_str)
        self.client_socket.sendto(json_str.encode('utf-8'), self.server_address)
        
        # 保存目标位置的z轴坐标
        if msg.points and len(msg.points) > 0 and len(msg.points[0].positions) >= 3:
            self.latest_target_z = msg.points[0].positions[2]

    def gripper_callback(self, msg):
        """处理夹爪控制指令"""
        command = msg.data
        if command == "open":
            open_gripper()
            self.get_logger().info('打开夹爪')
        elif command == "close":
            close_gripper()
            self.get_logger().info('闭合夹爪')
        else:
            self.get_logger().warn(f'未知的夹爪控制指令: {command}')

    def to_json(self, msg):
        data = {
            'joint_names': [joint_name for joint_name in msg.joint_names],
            'points': [{'positions': [_ for _ in point.positions],
                        'velocities': [_ for _ in point.velocities],
                        'accelerations': [_ for _ in point.accelerations]} for point in msg.points]
        }
        return json.dumps(data, default=str)
    
    def check_and_control_gripper(self, robot_info):
        """
        检查末端位置和目标位置的Z轴坐标，如果相近则控制夹爪
        """
        try:
            # 确保有目标位置和当前末端位置数据
            if self.latest_target_z is not None and len(robot_info.end_positions) >= 3:
                target_z = self.latest_target_z
                current_z = robot_info.end_positions[2]  # 第3位是z轴坐标
                print(f"当前末端位置Z轴坐标: {current_z:.6f}m, 目标位置Z轴坐标: {target_z:.6f}m, 差值：{abs(target_z - current_z) }")
                # 检查当前末端位置和目标位置是否相近（误差在5mm以内）
                if abs(target_z - current_z) < 0.005 and current_z > 0:
                    # 确保没有在该位置执行过夹爪动作
                    if self.last_gripper_z is None or abs(target_z - self.last_gripper_z) > 0.005:
                        # 更新已执行夹爪动作的位置
                        self.last_gripper_z = target_z
                        
                        # 执行夹爪控制
                        self.execute_gripper_control(target_z)
                    
        except Exception as e:
            self.get_logger().error(f'Error in check_and_control_gripper: {e}')
    
    def execute_gripper_control(self, target_z):
        """
        执行夹爪控制 - 先打开再闭合夹爪
        """
        try:
            # 打开夹爪
            open_gripper()
            
            # 等待一段时间让夹爪完全打开
            import time
            time.sleep(0.5)
            
            # 闭合夹爪进行抓取
            close_gripper()
            
            self.get_logger().info(f'Gripper control executed for target z: {target_z}')
        except Exception as e:
            self.get_logger().error(f'Error executing gripper control: {e}')

def main():
    rclpy.init()
    node = CodroidIO()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()