import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from grasp_interfaces.msg import GraspResult
from codroid_msgs.msg import RobotInfo
import time
import yaml
import os


class CodroidMoveTest(Node):
    def __init__(self):
        super().__init__('CodroidMoveTest')
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)

        # 发布者
        self.pub_move = self.create_publisher(JointTrajectory, 'RobotMove', qos)
        self.pub_grip = self.create_publisher(String, 'GripperControl', qos)
        self.pub_status = self.create_publisher(String, 'robot_status', qos)


        # 订阅者
        self.sub_grasp = self.create_subscription(
            GraspResult, '/grasp_result', self.grasp_result_callback, 10)
        self.sub_info = self.create_subscription(
            RobotInfo, '/RobotInfo', self.robot_info_callback, 10)

        # 加载补偿值配置
        self.compensation_dict = self.load_compensation_values()

        # 状态变量
        self.latest_grasp_result = None
        self.latest_robot_info = None
        self.target_position = None   # [x,y,z,rx,ry,rz]
        self.current_step = 0         # 0 未开始；1 旋转；2 移动；3 返回
        self.grasping_completed = False
        self.have_backed_sent = False  # 确保 "have backed" 只发送一次
        self.position_check_enabled = False  # 是否启用位置检测
        self.step_start_time = None   # 当前步骤开始时间

    # ---------- 加载补偿值 ----------
    def load_compensation_values(self):
        """从classes.yaml文件加载补偿值"""
        config_path = '/home/zyh/ZYH_WS/src/grasp_ws/config/classes.yaml'

        try:
            with open(config_path, 'r') as f:
                compensation_dict = yaml.safe_load(f)
            self.get_logger().info(f'成功加载补偿值配置，共{len(compensation_dict)}个类别')
            return compensation_dict
        except FileNotFoundError:
            self.get_logger().error(f'找不到配置文件: {config_path}')
            return {}
        except Exception as e:
            self.get_logger().error(f'加载配置文件时出错: {str(e)}')
            return {}

    # ---------- 回调 ----------
    def robot_info_callback(self, msg: RobotInfo):
        self.latest_robot_info = msg
        # 只有启用了位置检测并且有目标位置时才检查位置容差
        if self.position_check_enabled and self.target_position is not None:
            self.check_position_tolerance()

    def grasp_result_callback(self, msg: GraspResult):
        self.latest_grasp_result = msg
        self.get_logger().info(f'收到抓取结果: pos_base={msg.pos_base}, euler_base={msg.euler_base}')
        self.move_to_grasp_position()
        self.control_gripper("open")

    # ---------- 位置检查 ----------
    def check_position_tolerance(self):
        if self.target_position is None or self.latest_robot_info is None:
            return
        if len(self.latest_robot_info.end_positions) < 6:
            return

        curr = self.latest_robot_info.end_positions[:6]
        targ = self.target_position

        pos_diff = sum((curr[i] - targ[i]) ** 2 for i in range(3)) ** 0.5
        rot_diff = abs(curr[5] - targ[5])

        self.get_logger().debug(f'位置差值: {pos_diff:.6f}, 角度差值: {rot_diff:.6f}')

        # 根据不同步骤设置不同的容差标准
        pos_tolerance = 5  # 5mm
        rot_tolerance = 0.5   # 度
        
        # 对于第一步（只旋转），不需要检查位置
        pos_ok = True if self.current_step == 1 else pos_diff < pos_tolerance
        rot_ok = rot_diff < rot_tolerance

        # 添加超时机制，避免无限等待
        timeout =5.0 # 超时时间10秒
        elapsed_time = time.time() - self.step_start_time if self.step_start_time else 0

        # if (pos_ok and rot_ok) or elapsed_time > timeout:
        if (pos_ok and rot_ok) :
            # if elapsed_time > timeout:
            #     self.get_logger().warn(f'步骤 {self.current_step} 超时，强制进入下一步')
            #     self.get_logger().info(f'位置差值: {pos_diff:.6f}, 角度差值: {rot_diff:.6f}')
            # else:
            #     self.get_logger().info(f'步骤 {self.current_step} 完成，位置误差: {pos_diff:.6f}, 角度误差: {rot_diff:.6f}')
                
            self.position_check_enabled = False  # 暂时禁用位置检测
            
            if self.current_step == 1:
                self.get_logger().info('第一步（旋转）完成，执行第二步')
                self.execute_step_two()
            elif self.current_step == 2:
                self.get_logger().info('第二步（移动）完成，执行第三步')
                self.control_gripper("close")
                time.sleep(2)
                self.execute_step_three()
            elif self.current_step == 3:
                self.get_logger().info('第三步（返回放置位置）完成')
                self.control_gripper("open")
                time.sleep(2)
                self.execute_step_four()  # 执行第四步
            elif self.current_step == 4:
                self.get_logger().info('第四步（返回home点）完成')
                # 确保 "have backed" 只发送一次
                if not self.have_backed_sent:
                    self.publish_status("have backed")  # 发布状态信息
                    self.have_backed_sent = True

    # ---------- 状态信息发布 ----------
    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)
        self.get_logger().info(f'发布状态信息: {status}')

        
    # ---------- 夹爪控制 ----------
    def control_gripper(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.pub_grip.publish(msg)
        self.get_logger().info(f'发送夹爪控制命令: {cmd}')

    # ---------- 第一步：仅旋转 ----------
    def move_to_grasp_position(self):
        if self.latest_grasp_result is None:
            self.get_logger().warn('无抓取结果，无法移动')
            return

        self.grasping_completed = False
        self.have_backed_sent = False  # 重置标志，为下一次发送做准备
        self.step_start_time = time.time()  # 记录步骤开始时间
        pos_base = self.latest_grasp_result.pos_base
        euler_base = self.latest_grasp_result.euler_base

        msg = JointTrajectory()
        msg.joint_names = ["x", "y", "z", "rx", "ry", "rz"]
        p = JointTrajectoryPoint()

        # 保持当前 xyz，仅旋转 rz
        current_xyz = [-115.955, -320.591, -428.274]
        euler = [0.0, 0.0, float(euler_base[2])]
        p.positions = current_xyz + euler
        msg.points.append(p)

        self.pub_move.publish(msg)
        self.get_logger().info('发布第一步（旋转）消息')

        # 仅保存角度用于检查，位置部分不检查
        self.target_position = current_xyz+ euler
        self.current_step = 1
        self.position_check_enabled = True  # 启用位置检测

    # ---------- 第二步：平移+保持角度 ----------
    def execute_step_two(self):
        if self.latest_grasp_result is None:
            return

        self.step_start_time = time.time()  # 记录步骤开始时间
        pos_base = self.latest_grasp_result.pos_base
        euler_base = self.latest_grasp_result.euler_base
        cls_name = self.latest_grasp_result.cls_name

        # 从配置文件中获取对应类别的补偿值，默认为0
        z_compensation = self.compensation_dict.get(cls_name, 0.0)

        # 先进行X、Y轴运动，暂不改变Z轴
        msg1 = JointTrajectory()
        msg1.joint_names = ["x", "y", "z", "rx", "ry", "rz"]

        p1 = JointTrajectoryPoint()
        # X、Y轴移到目标位置，Z轴保持当前位置
        xy_move_positions = [float(pos_base[0]),
                             float(pos_base[1]),
                             -428.274]  # Z轴保持当前位置
        euler = [0.0, 0.0, float(euler_base[2])]
        p1.positions = xy_move_positions + euler
        from builtin_interfaces.msg import Duration
        p1.time_from_start = Duration(sec=1, nanosec=0)
        msg1.points.append(p1)

        self.pub_move.publish(msg1)
        self.get_logger().info('发布第二步第一阶段（X、Y轴移动）消息')

        # 等待X、Y轴移动完成
        time.sleep(2.0)
        
        # 再进行Z轴运动
        msg2 = JointTrajectory()
        msg2.joint_names = ["x", "y", "z", "rx", "ry", "rz"]
        
        p2 = JointTrajectoryPoint()
        adjusted_xyz = [float(pos_base[0]),
                        float(pos_base[1]),
                        float(pos_base[2]) + z_compensation]  # 下降到目标高度并加上补偿值
        p2.positions = adjusted_xyz + euler
        p2.time_from_start = Duration(sec=2, nanosec=0)
        msg2.points.append(p2)
        
        self.pub_move.publish(msg2)
        self.get_logger().info(f'发布第二步第二阶段（Z轴下降）消息，类别: {cls_name}，补偿值: {z_compensation}')

        self.target_position = list(adjusted_xyz) + euler
        self.current_step = 2
        self.position_check_enabled = True  # 启用位置检测
        
    # ---------- 第三步：返回初始 ----------
    def execute_step_three(self):
        # 获取当前位置
        if self.latest_robot_info is None or len(self.latest_robot_info.end_positions) < 6:
            self.get_logger().warn('无法获取机器人当前位置')
            return
            
        self.step_start_time = time.time()  # 记录步骤开始时间
        current_positions = self.latest_robot_info.end_positions[:6]
        self.get_logger().info(f'当前机器人位置: {current_positions}')
        
        # 先进行Z轴运动回到安全高度，X、Y保持当前位置
        msg1 = JointTrajectory()
        msg1.joint_names = ["x", "y", "z", "rx", "ry", "rz"]
        p1 = JointTrajectoryPoint()

        # 提升Z轴到安全高度，X、Y保持当前位置
        '''改高度用于第二问&第三问'''
        safe_z_height = -428.274  # 第三问的Z高度
        # safe_z_height = -0.10185  # 第二问的Z高度
        z_up_positions = [current_positions[0], current_positions[1], safe_z_height]
        throw_euler = [-0.199, -3.085, 1.215]
        p1.positions = z_up_positions + throw_euler
        from builtin_interfaces.msg import Duration
        p1.time_from_start = Duration(sec=1, nanosec=0)
        msg1.points.append(p1)

        self.pub_move.publish(msg1)
        self.get_logger().info(f'发布第三步第一阶段（Z轴提升）消息: {z_up_positions}')

        # 等待Z轴提升完成
        time.sleep(3.0)

        # 再进行X、Y轴运动到目标位置
        msg2 = JointTrajectory()
        msg2.joint_names = ["x", "y", "z", "rx", "ry", "rz"]
        p2 = JointTrajectoryPoint()

        '''改位置用于第二问&第三问'''
        throw_xyz = [588.896, -372.898, -205.981]  # 第三问置的目标位置
        # throw_xyz = [-0.140, 0.275, -0.10185]  # 第二问的目标位置
        p2.positions = throw_xyz + throw_euler
        p2.time_from_start = Duration(sec=2, nanosec=0)
        msg2.points.append(p2)

        self.pub_move.publish(msg2)
        self.get_logger().info(f'发布第三步第二阶段（X、Y轴移动）消息: {throw_xyz}')

        self.target_position = list(throw_xyz) + throw_euler
        self.current_step = 3
        self.position_check_enabled = True  # 启用位置检测
        
      # ---------- 第四步：返回home点 ----------
    def execute_step_four(self):
        self.step_start_time = time.time()  # 记录步骤开始时间
        msg = JointTrajectory()
        msg.joint_names = ["x", "y", "z", "rx", "ry", "rz"]
        p = JointTrajectoryPoint()

        home_xyz = [-115.955, -320.591, -428.274]
        home_euler = [-0.199, -3.085, 1.215]  # 修改为指定的角度
        p.positions = home_xyz + home_euler
        msg.points.append(p)

        self.pub_move.publish(msg)
        self.get_logger().info('发布第四步（返回home点）消息')

        self.target_position = list(home_xyz) + home_euler
        self.current_step = 4
        self.position_check_enabled = True  # 启用位置检测


def main():
    rclpy.init()
    node = CodroidMoveTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()