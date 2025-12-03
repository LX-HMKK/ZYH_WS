import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
import socket
import json
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from codroid_msgs.msg import RobotInfo

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

        self.timer = self.create_timer(0.04, self.timer_callback)

    def timer_callback(self):
        try:
            data, _ = self.server_socket.recvfrom(1024)  # 接收服务器响应
            json_data = json.loads(data.decode('utf-8'))
            print(json_data)
            robot_info = RobotInfo()
            robot_info.joint_positions = json_data["joint_positions"]
            robot_info.end_positions = json_data["end_positions"]
            robot_info.state = json_data["state"]
            robot_info.fault_flag = json_data["fault_flag"]
            self.publisher.publish(robot_info)
            print(robot_info)
        except:
            return

    def listener_callback(self, msg):
        json_str = self.to_json(msg)
        print(json_str)
        self.client_socket.sendto(json_str.encode('utf-8'), self.server_address)

    def to_json(self, msg):
        data = {
            'joint_names': [joint_name for joint_name in msg.joint_names],
            'points': [{'positions': [_ for _ in point.positions],
                        'velocities': [_ for _ in point.velocities],
                        'accelerations': [_ for _ in point.accelerations]} for point in msg.points]
        }
        return json.dumps(data, default=str)
    
def main():
    rclpy.init()
    node = CodroidIO()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
