import rclpy
from rclpy.node import Node
from enum import IntEnum, auto
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from arm_interfaces.msg import GraspResult
from arm_interfaces.msg import RobotInfo
from builtin_interfaces.msg import Duration
from arm_utils import get_workspace_root, load_yaml_config, reliable_qos
import os
import time


DEFAULT_CONFIG_PATH = f"{get_workspace_root()}/src/arm_control/config/motion_config.yaml"


class Step(IntEnum):
    """抓取-放置状态机步骤。"""
    IDLE = 0
    ROTATE = auto()
    MOVE_XY = auto()
    LOWER_Z = auto()
    GRIP_CLOSE = auto()
    LIFT_Z = auto()
    MOVE_PLACE = auto()
    GRIP_OPEN = auto()
    RETURN_HOME = auto()


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

        # 状态变量
        self.latest_grasp_result: GraspResult | None = None
        self.latest_robot_info: RobotInfo | None = None
        self.current_step = Step.IDLE
        self.step_start_time = 0.0
        self.target_position: list[float] | None = None
        self.busy = False

        # 控制循环：10 Hz
        self._control_timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info("MotionNode 初始化完成，等待 /grasp_result")

    # ---------- 配置加载 ----------
    def load_config(self, path: str) -> dict:
        """从 YAML 加载运动参数。"""
        cfg = load_yaml_config(path)
        self.get_logger().info(f"已加载运动配置：{path}")
        return cfg

    def _get_float_list(self, key: str, default: list | None = None) -> list[float]:
        return [float(v) for v in self.cfg.get(key, default or [])]

    # ---------- 回调 ----------
    def robot_info_callback(self, msg: RobotInfo):
        self.latest_robot_info = msg

    def grasp_result_callback(self, msg: GraspResult):
        if self.busy:
            self.get_logger().warn("当前正在执行抓取周期，忽略新的抓取结果")
            return

        self.latest_grasp_result = msg
        self.get_logger().info(
            f"收到抓取结果: pos_base={msg.pos_base}, euler_base={msg.euler_base}, cls={msg.cls_name}"
        )
        self._start_cycle()

    # ---------- 状态机控制 ----------
    def _control_loop(self):
        """10 Hz 控制循环：推进状态机。"""
        if self.current_step == Step.IDLE:
            return

        # 检查步骤超时
        timeout = self.cfg.get("step_timeout", 10.0)
        if time.time() - self.step_start_time > timeout:
            self.get_logger().error(f"步骤 {self.current_step.name} 超时，中止周期并返回 home")
            self._abort_cycle()
            return

        if self.current_step == Step.GRIP_CLOSE:
            if time.time() - self.step_start_time >= self.cfg.get("gripper_close_delay", 2.0):
                self._start_lift_z()
            return

        if self.current_step == Step.GRIP_OPEN:
            if time.time() - self.step_start_time >= self.cfg.get("gripper_open_delay", 2.0):
                self._start_return_home()
            return

        # 需要位置校验的步骤
        if self.target_position is None or self.latest_robot_info is None:
            return
        if len(self.latest_robot_info.end_positions) < 6:
            return

        if self._position_reached():
            if self.current_step == Step.ROTATE:
                self._start_move_xy()
            elif self.current_step == Step.MOVE_XY:
                self._start_lower_z()
            elif self.current_step == Step.LOWER_Z:
                self._start_grip_close()
            elif self.current_step == Step.LIFT_Z:
                self._start_move_place()
            elif self.current_step == Step.MOVE_PLACE:
                self._start_grip_open()
            elif self.current_step == Step.RETURN_HOME:
                self._finish_cycle()

    def _position_reached(self) -> bool:
        """判断当前位姿是否到达目标容差内。"""
        curr = self.latest_robot_info.end_positions[:6]
        targ = self.target_position

        pos_diff = sum((curr[i] - targ[i]) ** 2 for i in range(3)) ** 0.5
        rot_diff = abs(curr[5] - targ[5])

        pos_tol = self.cfg.get("position_tolerance", 5.0)
        rot_tol = self.cfg.get("rotation_tolerance", 0.5)

        # 旋转步骤只检查旋转
        if self.current_step == Step.ROTATE:
            return rot_diff < rot_tol
        return pos_diff < pos_tol and rot_diff < rot_tol

    # ---------- 周期阶段 ----------
    def _start_cycle(self):
        """开始新的抓取-放置周期。"""
        self.busy = True
        self._start_rotate()
        self.control_gripper("open")

    def _finish_cycle(self):
        """周期结束，返回 idle 并通知上游。"""
        self.get_logger().info("周期完成，返回 home 点")
        self.publish_status("have backed")
        self.current_step = Step.IDLE
        self.target_position = None
        self.busy = False

    def _abort_cycle(self):
        """超时或失败时返回 home 并结束周期。"""
        home_xyz = self._get_float_list("home_xyz")
        home_euler = self._get_float_list("home_euler")
        self._publish_move(home_xyz + home_euler)
        self.current_step = Step.RETURN_HOME
        self.step_start_time = time.time()
        self.target_position = home_xyz + home_euler
        self.busy = False  # 允许新周期，但会先回到 home

    def _start_rotate(self):
        """第一步：仅旋转到目标偏航角。"""
        if self.latest_grasp_result is None:
            return

        euler_base = self.latest_grasp_result.euler_base
        current_xyz = self._get_float_list("current_xyz")
        target = current_xyz + [0.0, 0.0, float(euler_base[2])]

        self._publish_move(target)
        self.current_step = Step.ROTATE
        self.step_start_time = time.time()
        self.target_position = target
        self.get_logger().info(f"步骤 ROTATE: 目标={target}")

    def _start_move_xy(self):
        """第二步第一阶段：XY 平面移动到目标位置，Z 保持当前。"""
        pos_base = self.latest_grasp_result.pos_base
        current_z = self._get_float_list("current_xyz")[2]
        euler_base = self.latest_grasp_result.euler_base
        target = [float(pos_base[0]), float(pos_base[1]), current_z, 0.0, 0.0, float(euler_base[2])]

        self._publish_move(target, time_from_start_sec=1)
        self.current_step = Step.MOVE_XY
        self.step_start_time = time.time()
        self.target_position = target
        self.get_logger().info(f"步骤 MOVE_XY: 目标={target}")

    def _start_lower_z(self):
        """第二步第二阶段：Z 轴下降并应用类别补偿。"""
        pos_base = self.latest_grasp_result.pos_base
        euler_base = self.latest_grasp_result.euler_base
        cls_name = self.latest_grasp_result.cls_name

        compensation = self.cfg.get("class_compensation", {}).get(cls_name, 0.0)
        target = [
            float(pos_base[0]),
            float(pos_base[1]),
            float(pos_base[2]) + float(compensation),
            0.0,
            0.0,
            float(euler_base[2]),
        ]

        self._publish_move(target, time_from_start_sec=2)
        self.current_step = Step.LOWER_Z
        self.step_start_time = time.time()
        self.target_position = target
        self.get_logger().info(f"步骤 LOWER_Z: 类别={cls_name}, 补偿={compensation}, 目标={target}")

    def _start_grip_close(self):
        """闭合夹爪并等待。"""
        self.control_gripper("close")
        self.current_step = Step.GRIP_CLOSE
        self.step_start_time = time.time()
        self.target_position = None
        self.get_logger().info("步骤 GRIP_CLOSE")

    def _start_lift_z(self):
        """第三步第一阶段：提升 Z 轴到安全高度。"""
        if self.latest_robot_info is None or len(self.latest_robot_info.end_positions) < 6:
            self.get_logger().warn("无法获取当前位置，跳过提升")
            return

        current = self.latest_robot_info.end_positions[:6]
        safe_z = self.cfg.get("safe_z_height")
        place_euler = self._get_float_list("place_euler")
        target = [current[0], current[1], safe_z] + place_euler

        self._publish_move(target, time_from_start_sec=1)
        self.current_step = Step.LIFT_Z
        self.step_start_time = time.time()
        self.target_position = target
        self.get_logger().info(f"步骤 LIFT_Z: 目标={target}")

    def _start_move_place(self):
        """第三步第二阶段：移动到放置位置。"""
        place_xyz = self._get_float_list("place_xyz")
        place_euler = self._get_float_list("place_euler")
        target = place_xyz + place_euler

        self._publish_move(target, time_from_start_sec=2)
        self.current_step = Step.MOVE_PLACE
        self.step_start_time = time.time()
        self.target_position = target
        self.get_logger().info(f"步骤 MOVE_PLACE: 目标={target}")

    def _start_grip_open(self):
        """打开夹爪并等待。"""
        self.control_gripper("open")
        self.current_step = Step.GRIP_OPEN
        self.step_start_time = time.time()
        self.target_position = None
        self.get_logger().info("步骤 GRIP_OPEN")

    def _start_return_home(self):
        """第四步：返回 home 点。"""
        home_xyz = self._get_float_list("home_xyz")
        home_euler = self._get_float_list("home_euler")
        target = home_xyz + home_euler

        self._publish_move(target)
        self.current_step = Step.RETURN_HOME
        self.step_start_time = time.time()
        self.target_position = target
        self.get_logger().info(f"步骤 RETURN_HOME: 目标={target}")

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
