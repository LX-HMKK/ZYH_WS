import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class CodroidMoveTest(Node):
    def __init__(self):
        super().__init__('CodroidMoveTest')
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.publisher_ = self.create_publisher(JointTrajectory, 'RobotMove', qos_profile)

        # time.sleep(1)

        self.test1()

    def test1(self):       
        msg = JointTrajectory()
        msg.joint_names = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
        p1 = JointTrajectoryPoint()
        p1.positions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        msg.points.append(p1)

        p2 = JointTrajectoryPoint()
        p2.positions = [0.0, 0.0, 1.5, 0.0, 1.5, 0.0]
        msg.points.append(p2)

        self.publisher_.publish(msg)
        self.get_logger().info(f'发布消息: "{msg}"')

    def test2(self):       
        msg = JointTrajectory()
        msg.joint_names = ["y", "z"]
        p1 = JointTrajectoryPoint()
        p1.positions = [0.2, 0.56]
        p1.velocities = [0.02, 0.02]
        msg.points.append(p1)

        p2 = JointTrajectoryPoint()
        p2.positions = [0.3, 0.36]
        p2.accelerations = [0.2, 0.2]
        msg.points.append(p2)

        self.publisher_.publish(msg)
        self.get_logger().info(f'发布消息: "{msg}"')

def main():
    rclpy.init()
    node = CodroidMoveTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
