import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from arm_interfaces.msg import GraspResult
from arm_interfaces.msg import RobotInfo
from builtin_interfaces.msg import Duration
from arm_utils import get_workspace_root, load_yaml_config, reliable_qos
from .motion_state_machine import PickPlaceStateMachine


DEFAULT_CONFIG_PATH = f"{get_workspace_root()}/src/arm_control/config/motion_config.yaml"


class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")
        qos = reliable_qos(depth=10)

        # 发布者
        self.pub_move = self.create_publisher(JointTrajectory, "RobotMove", qos)
        self.pub_grip = self.create_publisher(String, "GripperControl", qos)
        self.pub_status = self.create_publisher(String, "robot_status", qos)

        # 订阅者
        self.sub_grasp = self.create_subscription(
            GraspResult, "/grasp_result", self.grasp_result_callback, 10
        )
        self.sub_info = self.create_subscription(
            RobotInfo, "/RobotInfo", self.robot_info_callback, 10
        )

        # 加载配置
        self.declare_parameter("motion_config_path", DEFAULT_CONFIG_PATH)
        config_path = self.get_parameter("motion_config_path").value
        self.cfg = self.load_config(config_path)

        # 状态机
        self.state_machine = PickPlaceStateMachine(self.cfg, self.get_logger())

        # 状态变量
        self.latest_grasp_result: GraspResult | None = None
        self.latest_robot_info: RobotInfo | None = None

        # 控制循环：10 Hz
        self._control_timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info("MotionNode 初始化完成，等待 /grasp_result")

    # ---------- 配置加载 ----------
    def load_config(self, path: str) -> dict:
        """从 YAML 加载运动参数。"""
        cfg = load_yaml_config(path)
        self.get_logger().info(f"已加载运动配置：{path}")
        return cfg

    # ---------- 回调 ----------
    def robot_info_callback(self, msg: RobotInfo):
        self.latest_robot_info = msg

    def grasp_result_callback(self, msg: GraspResult):
        if self.state_machine.busy:
            self.get_logger().warn("当前正在执行抓取周期，忽略新的抓取结果")
            return

        self.latest_grasp_result = msg
        self.get_logger().info(
            f"收到抓取结果: pos_base={msg.pos_base}, euler_base={msg.euler_base}, cls={msg.cls_name}"
        )

    # ---------- 状态机控制 ----------
    def _control_loop(self):
        """10 Hz 控制循环：推进状态机。"""
        if self.latest_grasp_result is not None and not self.state_machine.busy:
            self.state_machine.start_cycle(self.latest_grasp_result)
            self.latest_grasp_result = None

        cmd = self.state_machine.tick(self.latest_robot_info)

        if cmd.target_position is not None:
            self._publish_move(cmd.target_position, cmd.time_from_start_sec)

        if cmd.gripper_cmd:
            self.control_gripper(cmd.gripper_cmd)

        if cmd.status:
            self.publish_status(cmd.status)

        if cmd.finished or cmd.aborted:
            self.latest_grasp_result = None

    # ---------- 工具方法 ----------
    def _publish_move(self, positions: list[float], time_from_start_sec: int = 0):
        """发布 JointTrajectory 消息。"""
        msg = JointTrajectory()
        msg.joint_names = ["x", "y", "z", "rx", "ry", "rz"]
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in positions]
        if time_from_start_sec > 0:
            p.time_from_start = Duration(sec=time_from_start_sec, nanosec=0)
        msg.points.append(p)
        self.pub_move.publish(msg)

    def control_gripper(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.pub_grip.publish(msg)
        self.get_logger().info(f"发送夹爪控制命令: {cmd}")

    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)
        self.get_logger().info(f"发布状态信息: {status}")


def main():
    rclpy.init()
    node = MotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
