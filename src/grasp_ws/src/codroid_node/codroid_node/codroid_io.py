import rclpy
import sys
import os
sys.path.append("/home/zyh/ZYH_WS/src/Gloria-M-SDK-1.0.0/motor")

from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
import socket
import json
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from codroid_msgs.msg import RobotInfo
import select

from DM_Motor_Test import *

class CodroidIO(Node):
    def __init__(self):
        super().__init__('CodroidIO')
        self.publisher = self.create_publisher(RobotInfo, '/RobotInfo', 10)
        # 订阅RobotMove消息
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.subscription = self.create_subscription(
            JointTrajectory, 
            'RobotMove', 
            self.robot_move_callback, 
            qos_profile)
        
        # 订阅夹爪控制话题
        self.gripper_subscription = self.create_subscription(
            String,
            'GripperControl',
            self.gripper_callback,
            10)
        
        # 初始化夹爪控制器
        init_gripper('/dev/ttyACM0')


        # 本机（上位机）作为服务器，监听固定IP和端口（下位机将主动连接此端口）
        self.server_addr = ('172.16.26.125', 10001)  # 上位机IP和固定端口
        self.server_socket = None
        self.downstream_conns = {}  # 存储所有下位机连接：key=(ip, port), value=socket对象
        self.recv_buffers = {}      # 每个下位机的独立接收缓冲区（避免数据混叠）
        self.init_server()

    def init_server(self):
        """初始化TCP服务器，监听上位机的固定端口，等待下位机（172.16.26.126）连接"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.server_addr)
        self.server_socket.listen(5)  # 允许最多5个下位机同时连接
        self.server_socket.setblocking(False)  # 非阻塞模式
        self.get_logger().info(f'上位机服务器启动，监听 {self.server_addr[0]}:{self.server_addr[1]}，等待下位机连接...')

    def robot_move_callback(self, msg):
        """处理RobotMove消息，发送给所有已连接的下位机"""
        try:
            if not msg.points:
                self.get_logger().warn('Received JointTrajectory with no points')
                return

            # 处理目标位置
            target_point = msg.points[-1]
            positions = [float(pos) for pos in target_point.positions]

            # 对前三位乘以10000，后三位除以57.3
            if len(positions) >= 6:
                transformed_positions = [
                    positions[0],
                    positions[1],
                    positions[2],
                    positions[3],
                    positions[4],
                    positions[5]
                ]
            elif len(positions) > 0:
                # 如果少于6位，则按比例处理
                transformed_positions = []
                for i, pos in enumerate(positions):
                    if i < 3:
                        transformed_positions.append(pos)
                    else:
                        transformed_positions.append(pos)
            
            # 构造6位位置字符串
            if len(positions) >= 6:
                position_str = ','.join([f'{pos:.6f}' for pos in positions[:6]])
            else:
                padded_positions = positions + [0.0] * (6 - len(positions))
                position_str = ','.join([f'{pos:.6f}' for pos in padded_positions])

            # 发送数据给所有已连接的下位机（无论其端口如何）
            send_data = f'[{position_str}]'.encode('ascii')
            for addr, conn in list(self.downstream_conns.items()):
                try:
                    conn.sendall(send_data)
                    self.get_logger().info(f'发送数据到下位机 {addr}：{position_str}')
                except Exception as e:
                    self.get_logger().error(f'向下位机 {addr} 发送失败：{e}')
                    # 移除失效连接
                    conn.close()
                    del self.downstream_conns[addr]
                    del self.recv_buffers[addr]

            if not self.downstream_conns:
                self.get_logger().warn('无已连接的下位机，无法发送数据')

        except Exception as e:
            self.get_logger().error(f'处理RobotMove消息出错：{e}')

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

    def run(self):
        """主循环：处理新连接、接收下位机数据、ROS事件"""
        while rclpy.ok():
            # 1. 检测新的下位机连接（允许下位机用随机端口连接）
            try:
                conn, addr = self.server_socket.accept()
                # 只接受来自下位机固定IP（172.16.26.126）的连接（可选，增强安全性）
                if addr[0] != '172.16.26.126':
                    self.get_logger().warning(f'拒绝非下位机IP连接：{addr}')
                    conn.close()
                    continue
                if addr not in self.downstream_conns:
                    conn.setblocking(False)
                    self.downstream_conns[addr] = conn
                    self.recv_buffers[addr] = b''  # 初始化该下位机的缓冲区
                    self.get_logger().info(f'下位机 {addr} 已连接（端口随机），当前连接数：{len(self.downstream_conns)}')
                else:
                    self.get_logger().warning(f'下位机 {addr} 重复连接，已忽略')
                    conn.close()
            except BlockingIOError:
                pass  # 无新连接，继续

            # 2. 接收所有下位机的数据（使用独立缓冲区）
            for addr in list(self.downstream_conns.keys()):
                conn = self.downstream_conns[addr]
                try:
                    data = conn.recv(1024)
                    if not data:
                        # 下位机断开连接
                        self.get_logger().info(f'下位机 {addr} 断开连接')
                        conn.close()
                        del self.downstream_conns[addr]
                        del self.recv_buffers[addr]
                        continue

                    # 按行解析数据（每个下位机单独缓存）
                    self.recv_buffers[addr] += data
                    lines = self.recv_buffers[addr].split(b'\n')
                    self.recv_buffers[addr] = lines.pop() if lines else b''

                    for line in lines:
                        if line:
                            raw_data = line.decode('utf-8').strip()
                            self.get_logger().debug(f'从下位机 {addr} 收到：{raw_data}')
                            self.process_position_data(raw_data)

                except BlockingIOError:
                    pass  # 无数据可读，继续
                except Exception as e:
                    self.get_logger().error(f'从下位机 {addr} 接收数据出错：{e}')
                    conn.close()
                    del self.downstream_conns[addr]
                    del self.recv_buffers[addr]

            # 3. 处理ROS事件
            rclpy.spin_once(self, timeout_sec=0.01)

    def process_position_data(self, data_string):
        """解析下位机发送的位置数据并发布"""
        try:
            prefix = 'get #real#6#'
            if data_string.startswith(prefix):
                data_string = data_string[len(prefix):].strip()
            
            num_parts = [x.strip() for x in data_string.split(',') if self.is_float(x.strip())]
            positions = [float(p) for p in num_parts]

            if len(positions) >= 6:
                robot_info = RobotInfo()
                robot_info.joint_positions = positions[:6]
                robot_info.end_positions = positions[:6]
                robot_info.state = "Normal"
                self.publisher.publish(robot_info)
                self.get_logger().info(f'发布6位位姿：{positions[:6]}')
            elif len(positions) > 0:
                self.get_logger().warning(f'数据不足6位：{positions}（原始数据：{data_string}）')

        except ValueError as e:
            self.get_logger().error(f'解析失败：{e}（原始数据：{data_string}）')

    def is_float(self, s):
        """判断字符串是否可转为浮点数"""
        try:
            float(s)
            return True
        except:
            return False

def main(args=None):
    rclpy.init(args=args)
    codroid_io = CodroidIO()
    try:
        codroid_io.run()
    except KeyboardInterrupt:
        pass
    finally:
        # 关闭所有下位机连接
        for conn in codroid_io.downstream_conns.values():
            conn.close()
        if codroid_io.server_socket:
            codroid_io.server_socket.close()
        codroid_io.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()