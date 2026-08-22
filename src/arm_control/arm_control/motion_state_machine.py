"""抓取-放置状态机纯 Python 实现。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum, auto
class Step(IntEnum):
    """抓取-放置状态机步骤。"""

    IDLE = 0
    WAIT_FOR_ROBOT_INFO = auto()
    ROTATE_TO_PICK = auto()
    LIFT_TO_PICK_SAFE = auto()
    MOVE_TO_PICK_XY = auto()
    LOWER_TO_PICK = auto()
    GRIP_CLOSE = auto()
    LIFT_TO_PLACE_SAFE = auto()
    ROTATE_TO_PLACE = auto()
    MOVE_TO_PLACE_XY = auto()
    LOWER_TO_PLACE = auto()
    GRIP_OPEN = auto()
    LIFT_TO_HOME_SAFE = auto()
    ROTATE_TO_HOME = auto()
    MOVE_TO_HOME_XY = auto()
    LOWER_TO_HOME = auto()


@dataclass
class StateMachineOutput:
    """状态机单次 tick 的输出命令。"""

    step: Step
    target_position: list[float] | None = None
    gripper_cmd: str | None = None
    status: str | None = None
    time_from_start_sec: int = 0
    finished: bool = False
    aborted: bool = False


class _NoopLogger:
    """无操作日志器，用于非 ROS 环境。"""

    def info(self, msg: str) -> None:
        pass

    def warn(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


class _SafePosePlanner:
    """将笛卡尔空间起点到终点的运动拆分为安全的四步序列。

    四步为：旋转（可选）、抬升/下降 Z 到安全高度、XY 平面移动、最终 Z 向接近。
    当 ``rotate_first=True`` 时，先在起点 Z 旋转，再抬升；否则先抬升，再在安全
    高度旋转。返回值中的姿态均为 [x, y, z, rx, ry, rz] 列表。
    """

    def __init__(self, safe_z: float):
        self.safe_z = float(safe_z)

    def segment(
        self, start: list[float], end: list[float], rotate_first: bool = True
    ) -> list[list[float]]:
        """规划一段安全的抓取/放置/归位轨迹。"""
        sx, sy, sz, srx, sry, srz = start
        ex, ey, ez, erx, ery, erz = end
        s_ori = [float(srx), float(sry), float(srz)]
        e_ori = [float(erx), float(ery), float(erz)]

        poses: list[list[float]] = []
        if rotate_first:
            # 1) 在起点 Z 完成姿态调整
            poses.append([float(sx), float(sy), float(sz)] + e_ori)
            # 2) 抬升至安全高度（保持目标姿态）
            poses.append([float(sx), float(sy), self.safe_z] + e_ori)
        else:
            # 1) 先抬升至安全高度（保持当前姿态）
            poses.append([float(sx), float(sy), self.safe_z] + s_ori)
            # 2) 在安全高度完成姿态调整
            poses.append([float(sx), float(sy), self.safe_z] + e_ori)

        # 3) 在安全高度做 XY 平面移动
        poses.append([float(ex), float(ey), self.safe_z] + e_ori)
        # 4) 下降至终点 Z
        poses.append([float(ex), float(ey), float(ez)] + e_ori)
        return poses


class PickPlaceStateMachine:
    """抓取-放置状态机。"""

    _ROTATION_ONLY_STEPS = {
        Step.ROTATE_TO_PICK,
        Step.ROTATE_TO_PLACE,
        Step.ROTATE_TO_HOME,
    }

    def __init__(self, cfg: dict, logger=None):
        self.cfg = cfg
        self.logger = logger or _NoopLogger()
        self._planner = _SafePosePlanner(cfg.get("safe_z_height", 0.0))

        self.current_step = Step.IDLE
        self.step_start_time = 0.0
        self.target_position: list[float] | None = None
        self.busy = False

        self.grasp_result = None
        self.latest_robot_info = None
        self._pending_output: StateMachineOutput | None = None

        self._pick_target: list[float] | None = None
        self._place_target: list[float] | None = None
        self._home_target: list[float] | None = None

    # ---------- 周期控制 ----------
    def start_cycle(self, grasp_result) -> None:
        """开始新的抓取-放置周期。"""
        self.grasp_result = grasp_result
        self.busy = True

        self._pick_target = self._build_pick_target()
        self._place_target = self._get_float_list("place_xyz") + self._get_float_list(
            "place_euler"
        )
        self._home_target = self._get_float_list("home_xyz") + self._get_float_list(
            "home_euler"
        )

        if self._current_robot_pose() is None:
            self.current_step = Step.WAIT_FOR_ROBOT_INFO
            self.step_start_time = time.monotonic()
            self.target_position = None
            self.logger.info("等待 /RobotInfo 以规划安全轨迹...")
            return

        output = self._start_rotate_to_pick()
        output.gripper_cmd = "open"
        self._pending_output = output

    def tick(self, robot_info) -> StateMachineOutput:
        """推进状态机并返回本次需要执行的命令。"""
        self.latest_robot_info = robot_info

        if self._pending_output is not None:
            out = self._pending_output
            self._pending_output = None
            return out

        if self.current_step == Step.IDLE:
            return StateMachineOutput(step=Step.IDLE)

        if self.current_step == Step.WAIT_FOR_ROBOT_INFO:
            current = self._current_robot_pose()
            if current is not None:
                self.logger.info("已获取 /RobotInfo，开始规划安全轨迹")
                output = self._start_rotate_to_pick()
                output.gripper_cmd = "open"
                return output
            return StateMachineOutput(step=Step.WAIT_FOR_ROBOT_INFO)

        # 检查步骤超时
        timeout = self.cfg.get("step_timeout", 10.0)
        if time.monotonic() - self.step_start_time > timeout:
            self.logger.error(f"步骤 {self.current_step.name} 超时，中止周期并返回 home")
            return self._abort_cycle()

        if self.current_step == Step.GRIP_CLOSE:
            if time.monotonic() - self.step_start_time >= self.cfg.get(
                "gripper_close_delay", 2.0
            ):
                return self._start_lift_to_place_safe()
            return StateMachineOutput(step=Step.GRIP_CLOSE)

        if self.current_step == Step.GRIP_OPEN:
            if time.monotonic() - self.step_start_time >= self.cfg.get(
                "gripper_open_delay", 2.0
            ):
                return self._start_lift_to_home_safe()
            return StateMachineOutput(step=Step.GRIP_OPEN)

        # 需要位置校验的步骤
        if self.target_position is None or robot_info is None:
            return StateMachineOutput(step=self.current_step)
        if len(robot_info.end_positions) < 6:
            return StateMachineOutput(step=self.current_step)

        if self._position_reached():
            if self.current_step == Step.ROTATE_TO_PICK:
                return self._start_lift_to_pick_safe()
            elif self.current_step == Step.LIFT_TO_PICK_SAFE:
                return self._start_move_to_pick_xy()
            elif self.current_step == Step.MOVE_TO_PICK_XY:
                return self._start_lower_to_pick()
            elif self.current_step == Step.LOWER_TO_PICK:
                return self._start_grip_close()
            elif self.current_step == Step.LIFT_TO_PLACE_SAFE:
                return self._start_rotate_to_place()
            elif self.current_step == Step.ROTATE_TO_PLACE:
                return self._start_move_to_place_xy()
            elif self.current_step == Step.MOVE_TO_PLACE_XY:
                return self._start_lower_to_place()
            elif self.current_step == Step.LOWER_TO_PLACE:
                return self._start_grip_open()
            elif self.current_step == Step.LIFT_TO_HOME_SAFE:
                return self._start_rotate_to_home()
            elif self.current_step == Step.ROTATE_TO_HOME:
                return self._start_move_to_home_xy()
            elif self.current_step == Step.MOVE_TO_HOME_XY:
                return self._start_lower_to_home()
            elif self.current_step == Step.LOWER_TO_HOME:
                return self._finish_cycle()

        return StateMachineOutput(step=self.current_step)

    # ---------- 工具方法 ----------
    def _get_float_list(self, key: str, default: list | None = None) -> list[float]:
        return [float(v) for v in self.cfg.get(key, default or [])]

    def _build_pick_target(self) -> list[float]:
        """根据抓取结果构建目标拾取位姿 [x, y, z, rx, ry, rz]。"""
        pos_base = self.grasp_result.pos_base
        euler_base = self.grasp_result.euler_base
        cls_name = self.grasp_result.cls_name

        compensation = self.cfg.get("class_compensation", {}).get(cls_name, 0.0)
        return [
            float(pos_base[0]),
            float(pos_base[1]),
            float(pos_base[2]) + float(compensation),
            0.0,
            0.0,
            float(euler_base[2]),
        ]

    def _get_home_target(self) -> list[float]:
        if self._home_target is None:
            self._home_target = self._get_float_list(
                "home_xyz"
            ) + self._get_float_list("home_euler")
        return self._home_target

    def _current_robot_pose(self) -> list[float] | None:
        if self.latest_robot_info is None or len(self.latest_robot_info.end_positions) < 6:
            return None
        return [float(v) for v in self.latest_robot_info.end_positions[:6]]

    def _position_reached(self) -> bool:
        """判断当前位姿是否到达目标容差内。"""
        curr = self.latest_robot_info.end_positions[:6]
        targ = self.target_position

        pos_diff = sum((curr[i] - targ[i]) ** 2 for i in range(3)) ** 0.5
        rot_diff = max(
            self._angle_diff(curr[i], targ[i]) for i in range(3, 6)
        )

        pos_tol = self.cfg.get("position_tolerance", 5.0)
        rot_tol = self.cfg.get("rotation_tolerance", 0.5)

        # 纯旋转步骤只检查旋转
        if self.current_step in self._ROTATION_ONLY_STEPS:
            return rot_diff < rot_tol
        return pos_diff < pos_tol and rot_diff < rot_tol

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """计算两个角度（度）之间的最小差值，处理 360° 环绕。"""
        diff = (a - b) % 360.0
        if diff > 180.0:
            diff -= 360.0
        return abs(diff)

    # ---------- 周期阶段 ----------
    def _finish_cycle(self):
        """周期结束，返回 idle 并通知上游。"""
        self.logger.info("周期完成，返回 home 点")
        self.current_step = Step.IDLE
        self.target_position = None
        self.busy = False
        return StateMachineOutput(step=Step.IDLE, status="have backed", finished=True)

    def _abort_cycle(self):
        """超时或失败时安全返回 home 并结束周期。"""
        home_target = self._get_home_target()
        current = self._current_robot_pose()

        if current is None:
            # 无法获取当前位置，退化为直接返回 home
            self.logger.error("无法获取当前位置，直接返回 home")
            self.current_step = Step.LOWER_TO_HOME
            self.step_start_time = time.monotonic()
            self.target_position = home_target
            self.busy = False
            return StateMachineOutput(
                step=Step.LOWER_TO_HOME,
                target_position=home_target,
                time_from_start_sec=2,
                aborted=True,
            )

        # 安全返回 home：先抬升，再旋转，再移动 XY，最后下降
        plan = self._planner.segment(current, home_target, rotate_first=False)
        self.current_step = Step.LIFT_TO_HOME_SAFE
        self.step_start_time = time.monotonic()
        self.target_position = plan[0]
        self.busy = True
        self.logger.error("中止周期，安全返回 home")
        return StateMachineOutput(
            step=Step.LIFT_TO_HOME_SAFE,
            target_position=plan[0],
            time_from_start_sec=0,
            aborted=True,
        )

    def _start_rotate_to_pick(self):
        """第一步：在当前 Z 高度旋转到目标拾取偏航角。"""
        start_pose = self._current_robot_pose()
        if start_pose is None:
            start_pose = self._get_float_list("current_xyz") + [0.0, 0.0, 0.0]
            self.logger.warn("缺少 /RobotInfo，使用 current_xyz 作为起点规划")
        plan = self._planner.segment(start_pose, self._pick_target, rotate_first=True)
        target = plan[0]

        self.current_step = Step.ROTATE_TO_PICK
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 ROTATE_TO_PICK: 目标={target}")
        return StateMachineOutput(step=Step.ROTATE_TO_PICK, target_position=target)

    def _start_lift_to_pick_safe(self):
        """抬升 Z 轴到安全高度，保持拾取姿态。"""
        start_pose = self._current_robot_pose()
        if start_pose is None:
            start_pose = self._get_float_list("current_xyz") + [0.0, 0.0, 0.0]
        plan = self._planner.segment(start_pose, self._pick_target, rotate_first=True)
        target = plan[1]

        self.current_step = Step.LIFT_TO_PICK_SAFE
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 LIFT_TO_PICK_SAFE: 目标={target}")
        return StateMachineOutput(
            step=Step.LIFT_TO_PICK_SAFE, target_position=target, time_from_start_sec=0
        )

    def _start_move_to_pick_xy(self):
        """在安全高度移动到拾取 XY 位置，保持拾取姿态。"""
        start_pose = self._current_robot_pose()
        if start_pose is None:
            start_pose = self._get_float_list("current_xyz") + [0.0, 0.0, 0.0]
        plan = self._planner.segment(start_pose, self._pick_target, rotate_first=True)
        target = plan[2]

        self.current_step = Step.MOVE_TO_PICK_XY
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 MOVE_TO_PICK_XY: 目标={target}")
        return StateMachineOutput(
            step=Step.MOVE_TO_PICK_XY, target_position=target, time_from_start_sec=1
        )

    def _start_lower_to_pick(self):
        """下降 Z 轴到拾取高度并应用类别补偿。"""
        target = self._pick_target

        self.current_step = Step.LOWER_TO_PICK
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(
            f"步骤 LOWER_TO_PICK: 类别={self.grasp_result.cls_name}, 目标={target}"
        )
        return StateMachineOutput(
            step=Step.LOWER_TO_PICK, target_position=target, time_from_start_sec=2
        )

    def _start_grip_close(self):
        """闭合夹爪并等待。"""
        self.current_step = Step.GRIP_CLOSE
        self.step_start_time = time.monotonic()
        self.target_position = None
        self.logger.info("步骤 GRIP_CLOSE")
        return StateMachineOutput(step=Step.GRIP_CLOSE, gripper_cmd="close")

    def _start_lift_to_place_safe(self):
        """从拾取点抬升 Z 轴到安全高度，保持拾取姿态。"""
        current = self._current_robot_pose()
        if current is None:
            self.logger.warn("无法获取当前位置，跳过抬升")
            return StateMachineOutput(step=Step.GRIP_CLOSE)

        plan = self._planner.segment(current, self._place_target, rotate_first=False)
        target = plan[0]

        self.current_step = Step.LIFT_TO_PLACE_SAFE
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 LIFT_TO_PLACE_SAFE: 目标={target}")
        return StateMachineOutput(
            step=Step.LIFT_TO_PLACE_SAFE, target_position=target, time_from_start_sec=0
        )

    def _start_rotate_to_place(self):
        """在安全高度旋转到放置姿态。"""
        current = self._current_robot_pose()
        if current is None:
            self.logger.warn("无法获取当前位置，跳过旋转")
            return StateMachineOutput(step=Step.LIFT_TO_PLACE_SAFE)

        plan = self._planner.segment(current, self._place_target, rotate_first=False)
        target = plan[1]

        self.current_step = Step.ROTATE_TO_PLACE
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 ROTATE_TO_PLACE: 目标={target}")
        return StateMachineOutput(
            step=Step.ROTATE_TO_PLACE, target_position=target, time_from_start_sec=0
        )

    def _start_move_to_place_xy(self):
        """在安全高度移动到放置 XY 位置，保持放置姿态。"""
        current = self._current_robot_pose()
        if current is None:
            self.logger.warn("无法获取当前位置，跳过 XY 移动")
            return StateMachineOutput(step=Step.ROTATE_TO_PLACE)

        plan = self._planner.segment(current, self._place_target, rotate_first=False)
        target = plan[2]

        self.current_step = Step.MOVE_TO_PLACE_XY
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 MOVE_TO_PLACE_XY: 目标={target}")
        return StateMachineOutput(
            step=Step.MOVE_TO_PLACE_XY, target_position=target, time_from_start_sec=1
        )

    def _start_lower_to_place(self):
        """下降 Z 轴到放置高度，保持放置姿态。"""
        target = self._place_target

        self.current_step = Step.LOWER_TO_PLACE
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 LOWER_TO_PLACE: 目标={target}")
        return StateMachineOutput(
            step=Step.LOWER_TO_PLACE, target_position=target, time_from_start_sec=2
        )

    def _start_grip_open(self):
        """打开夹爪并等待。"""
        self.current_step = Step.GRIP_OPEN
        self.step_start_time = time.monotonic()
        self.target_position = None
        self.logger.info("步骤 GRIP_OPEN")
        return StateMachineOutput(step=Step.GRIP_OPEN, gripper_cmd="open")

    def _start_lift_to_home_safe(self):
        """从放置点抬升 Z 轴到安全高度，保持放置姿态。"""
        current = self._current_robot_pose()
        if current is None:
            self.logger.warn("无法获取当前位置，跳过抬升")
            return StateMachineOutput(step=Step.GRIP_OPEN)

        home_target = self._get_home_target()
        plan = self._planner.segment(current, home_target, rotate_first=False)
        target = plan[0]

        self.current_step = Step.LIFT_TO_HOME_SAFE
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 LIFT_TO_HOME_SAFE: 目标={target}")
        return StateMachineOutput(
            step=Step.LIFT_TO_HOME_SAFE, target_position=target, time_from_start_sec=0
        )

    def _start_rotate_to_home(self):
        """在安全高度旋转到 home 姿态。"""
        current = self._current_robot_pose()
        if current is None:
            self.logger.warn("无法获取当前位置，跳过旋转")
            return StateMachineOutput(step=Step.LIFT_TO_HOME_SAFE)

        home_target = self._get_home_target()
        plan = self._planner.segment(current, home_target, rotate_first=False)
        target = plan[1]

        self.current_step = Step.ROTATE_TO_HOME
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 ROTATE_TO_HOME: 目标={target}")
        return StateMachineOutput(
            step=Step.ROTATE_TO_HOME, target_position=target, time_from_start_sec=0
        )

    def _start_move_to_home_xy(self):
        """在安全高度移动到 home XY 位置，保持 home 姿态。"""
        current = self._current_robot_pose()
        if current is None:
            self.logger.warn("无法获取当前位置，跳过 XY 移动")
            return StateMachineOutput(step=Step.ROTATE_TO_HOME)

        home_target = self._get_home_target()
        plan = self._planner.segment(current, home_target, rotate_first=False)
        target = plan[2]

        self.current_step = Step.MOVE_TO_HOME_XY
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 MOVE_TO_HOME_XY: 目标={target}")
        return StateMachineOutput(
            step=Step.MOVE_TO_HOME_XY, target_position=target, time_from_start_sec=1
        )

    def _start_lower_to_home(self):
        """下降 Z 轴到 home 高度，保持 home 姿态。"""
        target = self._get_home_target()

        self.current_step = Step.LOWER_TO_HOME
        self.step_start_time = time.monotonic()
        self.target_position = target
        self.logger.info(f"步骤 LOWER_TO_HOME: 目标={target}")
        return StateMachineOutput(
            step=Step.LOWER_TO_HOME, target_position=target, time_from_start_sec=2
        )
