"""抓取-放置状态机纯 Python 实现。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum, auto


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


class PickPlaceStateMachine:
    """抓取-放置状态机。"""

    def __init__(self, cfg: dict, logger=None):
        self.cfg = cfg
        self.logger = logger or _NoopLogger()

        self.current_step = Step.IDLE
        self.step_start_time = 0.0
        self.target_position: list[float] | None = None
        self.busy = False

        self.grasp_result = None
        self.latest_robot_info = None
        self._pending_output: StateMachineOutput | None = None

    # ---------- 周期控制 ----------
    def start_cycle(self, grasp_result) -> None:
        """开始新的抓取-放置周期。"""
        self.grasp_result = grasp_result
        self.busy = True
        output = self._start_rotate()
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

        # 检查步骤超时
        timeout = self.cfg.get("step_timeout", 10.0)
        if time.time() - self.step_start_time > timeout:
            self.logger.error(f"步骤 {self.current_step.name} 超时，中止周期并返回 home")
            return self._abort_cycle()

        if self.current_step == Step.GRIP_CLOSE:
            if time.time() - self.step_start_time >= self.cfg.get("gripper_close_delay", 2.0):
                return self._start_lift_z()
            return StateMachineOutput(step=Step.GRIP_CLOSE)

        if self.current_step == Step.GRIP_OPEN:
            if time.time() - self.step_start_time >= self.cfg.get("gripper_open_delay", 2.0):
                return self._start_return_home()
            return StateMachineOutput(step=Step.GRIP_OPEN)

        # 需要位置校验的步骤
        if self.target_position is None or robot_info is None:
            return StateMachineOutput(step=self.current_step)
        if len(robot_info.end_positions) < 6:
            return StateMachineOutput(step=self.current_step)

        if self._position_reached():
            if self.current_step == Step.ROTATE:
                return self._start_move_xy()
            elif self.current_step == Step.MOVE_XY:
                return self._start_lower_z()
            elif self.current_step == Step.LOWER_Z:
                return self._start_grip_close()
            elif self.current_step == Step.LIFT_Z:
                return self._start_move_place()
            elif self.current_step == Step.MOVE_PLACE:
                return self._start_grip_open()
            elif self.current_step == Step.RETURN_HOME:
                return self._finish_cycle()

        return StateMachineOutput(step=self.current_step)

    # ---------- 工具方法 ----------
    def _get_float_list(self, key: str, default: list | None = None) -> list[float]:
        return [float(v) for v in self.cfg.get(key, default or [])]

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
    def _finish_cycle(self):
        """周期结束，返回 idle 并通知上游。"""
        self.logger.info("周期完成，返回 home 点")
        self.current_step = Step.IDLE
        self.target_position = None
        self.busy = False
        return StateMachineOutput(step=Step.IDLE, status="have backed", finished=True)

    def _abort_cycle(self):
        """超时或失败时返回 home 并结束周期。"""
        home_xyz = self._get_float_list("home_xyz")
        home_euler = self._get_float_list("home_euler")
        target = home_xyz + home_euler

        self.current_step = Step.RETURN_HOME
        self.step_start_time = time.time()
        self.target_position = target
        self.busy = False  # 允许新周期，但会先回到 home
        return StateMachineOutput(
            step=Step.RETURN_HOME,
            target_position=target,
            time_from_start_sec=0,
            aborted=True,
        )

    def _start_rotate(self):
        """第一步：仅旋转到目标偏航角。"""
        euler_base = self.grasp_result.euler_base
        current_xyz = self._get_float_list("current_xyz")
        target = current_xyz + [0.0, 0.0, float(euler_base[2])]

        self.current_step = Step.ROTATE
        self.step_start_time = time.time()
        self.target_position = target
        self.logger.info(f"步骤 ROTATE: 目标={target}")
        return StateMachineOutput(step=Step.ROTATE, target_position=target)

    def _start_move_xy(self):
        """第二步第一阶段：XY 平面移动到目标位置，Z 保持当前。"""
        pos_base = self.grasp_result.pos_base
        current_z = self._get_float_list("current_xyz")[2]
        euler_base = self.grasp_result.euler_base
        target = [
            float(pos_base[0]),
            float(pos_base[1]),
            current_z,
            0.0,
            0.0,
            float(euler_base[2]),
        ]

        self.current_step = Step.MOVE_XY
        self.step_start_time = time.time()
        self.target_position = target
        self.logger.info(f"步骤 MOVE_XY: 目标={target}")
        return StateMachineOutput(
            step=Step.MOVE_XY, target_position=target, time_from_start_sec=1
        )

    def _start_lower_z(self):
        """第二步第二阶段：Z 轴下降并应用类别补偿。"""
        pos_base = self.grasp_result.pos_base
        euler_base = self.grasp_result.euler_base
        cls_name = self.grasp_result.cls_name

        compensation = self.cfg.get("class_compensation", {}).get(cls_name, 0.0)
        target = [
            float(pos_base[0]),
            float(pos_base[1]),
            float(pos_base[2]) + float(compensation),
            0.0,
            0.0,
            float(euler_base[2]),
        ]

        self.current_step = Step.LOWER_Z
        self.step_start_time = time.time()
        self.target_position = target
        self.logger.info(f"步骤 LOWER_Z: 类别={cls_name}, 补偿={compensation}, 目标={target}")
        return StateMachineOutput(
            step=Step.LOWER_Z, target_position=target, time_from_start_sec=2
        )

    def _start_grip_close(self):
        """闭合夹爪并等待。"""
        self.current_step = Step.GRIP_CLOSE
        self.step_start_time = time.time()
        self.target_position = None
        self.logger.info("步骤 GRIP_CLOSE")
        return StateMachineOutput(step=Step.GRIP_CLOSE, gripper_cmd="close")

    def _start_lift_z(self):
        """第三步第一阶段：提升 Z 轴到安全高度。"""
        if self.latest_robot_info is None or len(self.latest_robot_info.end_positions) < 6:
            self.logger.warn("无法获取当前位置，跳过提升")
            return StateMachineOutput(step=Step.GRIP_CLOSE)

        current = self.latest_robot_info.end_positions[:6]
        safe_z = self.cfg.get("safe_z_height")
        place_euler = self._get_float_list("place_euler")
        target = [current[0], current[1], safe_z] + place_euler

        self.current_step = Step.LIFT_Z
        self.step_start_time = time.time()
        self.target_position = target
        self.logger.info(f"步骤 LIFT_Z: 目标={target}")
        return StateMachineOutput(step=Step.LIFT_Z, target_position=target, time_from_start_sec=1)

    def _start_move_place(self):
        """第三步第二阶段：移动到放置位置。"""
        place_xyz = self._get_float_list("place_xyz")
        place_euler = self._get_float_list("place_euler")
        target = place_xyz + place_euler

        self.current_step = Step.MOVE_PLACE
        self.step_start_time = time.time()
        self.target_position = target
        self.logger.info(f"步骤 MOVE_PLACE: 目标={target}")
        return StateMachineOutput(
            step=Step.MOVE_PLACE, target_position=target, time_from_start_sec=2
        )

    def _start_grip_open(self):
        """打开夹爪并等待。"""
        self.current_step = Step.GRIP_OPEN
        self.step_start_time = time.time()
        self.target_position = None
        self.logger.info("步骤 GRIP_OPEN")
        return StateMachineOutput(step=Step.GRIP_OPEN, gripper_cmd="open")

    def _start_return_home(self):
        """第四步：返回 home 点。"""
        home_xyz = self._get_float_list("home_xyz")
        home_euler = self._get_float_list("home_euler")
        target = home_xyz + home_euler

        self.current_step = Step.RETURN_HOME
        self.step_start_time = time.time()
        self.target_position = target
        self.logger.info(f"步骤 RETURN_HOME: 目标={target}")
        return StateMachineOutput(step=Step.RETURN_HOME, target_position=target)
