import rclpy
import socket
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from robot_arm_interfaces.msg import RobotInfo
from robot_arm_utils import is_float, reliable_qos


DEFAULT_GRIPPER_PORT = "/dev/ttyACM0"
DEFAULT_SERVER_HOST = "172.16.26.125"
DEFAULT_SERVER_PORT = 10001
DEFAULT_ALLOWED_CLIENT = "172.16.26.126"


class RobotTcpServer:
    """上位机 TCP 服务器：接受下位机连接并转发运动指令。"""

    def __init__(self, host: str, port: int, allowed_client: str, logger):
        self.host = host
        self.port = port
        self.allowed_client = allowed_client
        self.logger = logger

        self.server_socket: socket.socket | None = None
        self.connections: dict[tuple[str, int], socket.socket] = {}
        self.buffers: dict[tuple[str, int], bytes] = {}

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.setblocking(False)
        self.logger.info(f"TCP 服务器监听 {self.host}:{self.port}")

    def stop(self):
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()
        self.buffers.clear()
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None

    def send_to_all(self, data: bytes):
        """向所有已连接的下位机发送数据，自动清理失效连接。"""
        dead = []
        for addr, conn in self.connections.items():
            try:
                conn.sendall(data)
                self.logger.info(f"发送数据到下位机 {addr}")
            except Exception as e:
                self.logger.error(f"向下位机 {addr} 发送失败：{e}")
                conn.close()
                dead.append(addr)

        for addr in dead:
            self.connections.pop(addr, None)
            self.buffers.pop(addr, None)

        if not self.connections:
            self.logger.warn("无已连接的下位机，无法发送数据")

    def tick(self) -> list[str]:
        """单轮处理：新连接、接收数据。返回解析到的数据行列表。"""
        lines: list[str] = []

        # 1. 接受新连接
        try:
            conn, addr = self.server_socket.accept()
            if addr[0] != self.allowed_client:
                self.logger.warning(f"拒绝非允许客户端连接：{addr}")
                conn.close()
            elif addr in self.connections:
                self.logger.warning(f"客户端 {addr} 重复连接，已忽略")
                conn.close()
            else:
                conn.setblocking(False)
                self.connections[addr] = conn
                self.buffers[addr] = b""
                self.logger.info(f"客户端 {addr} 已连接，当前连接数：{len(self.connections)}")
        except BlockingIOError:
            pass
        except Exception as e:
            self.logger.error(f"接受连接时出错：{e}")

        # 2. 接收数据
        dead = []
        for addr in list(self.connections.keys()):
            conn = self.connections[addr]
            try:
                data = conn.recv(1024)
                if not data:
                    self.logger.info(f"客户端 {addr} 断开连接")
                    conn.close()
                    dead.append(addr)
                    continue

                self.buffers[addr] += data
                buf_lines = self.buffers[addr].split(b"\n")
                self.buffers[addr] = buf_lines.pop() if buf_lines else b""
                lines.extend(line.decode("utf-8").strip() for line in buf_lines if line)
            except BlockingIOError:
                pass
            except Exception as e:
                self.logger.error(f"从 {addr} 接收数据出错：{e}")
                conn.close()
                dead.append(addr)

        for addr in dead:
            self.connections.pop(addr, None)
            self.buffers.pop(addr, None)

        return lines


class GripperController:
    """夹爪串口控制器封装，支持初始化失败后的重连。"""

    def __init__(self, port: str, logger):
        self.port = port
        self.logger = logger
        self._motor_module = None
        self._initialized = False

    def _load_sdk(self):
        """加载内联的夹爪驱动模块。"""
        if self._motor_module is None:
            from robot_arm_control import gripper_driver as motor_module

            self._motor_module = motor_module

    def initialize(self) -> bool:
        """尝试初始化夹爪。失败时记录日志，不抛异常。"""
        try:
            self._load_sdk()
            self._motor_module.init_gripper(self.port)
            self._initialized = True
            self.logger.info(f"夹爪初始化成功：{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"夹爪初始化失败：{e}")
            self._initialized = False
            return False

    def open(self):
        if not self._initialized:
            self.logger.warn("夹爪未初始化，跳过 open 命令")
            return
        try:
            self._motor_module.open_gripper()
            self.logger.info("打开夹爪")
        except Exception as e:
            self.logger.error(f"打开夹爪失败：{e}")

    def close(self):
        if not self._initialized:
            self.logger.warn("夹爪未初始化，跳过 close 命令")
            return
        try:
            self._motor_module.close_gripper()
            self.logger.info("闭合夹爪")
        except Exception as e:
            self.logger.error(f"闭合夹爪失败：{e}")


class IONode(Node):
    """上位机 IO 节点：聚合 TCP 下位机通信、夹爪控制与机器人状态发布。"""

    def __init__(self):
        super().__init__("robot_arm_io")

        # 参数声明
        self.declare_parameter("server_host", DEFAULT_SERVER_HOST)
        self.declare_parameter("server_port", DEFAULT_SERVER_PORT)
        self.declare_parameter("allowed_client", DEFAULT_ALLOWED_CLIENT)
        self.declare_parameter("gripper_port", DEFAULT_GRIPPER_PORT)
        self.declare_parameter("auto_init_gripper", True)

        server_host = self.get_parameter("server_host").value
        server_port = self.get_parameter("server_port").value
        allowed_client = self.get_parameter("allowed_client").value
        gripper_port = self.get_parameter("gripper_port").value
        auto_init = self.get_parameter("auto_init_gripper").value

        # 发布者
        self.publisher = self.create_publisher(RobotInfo, "/RobotInfo", 10)

        # 订阅者
        qos = reliable_qos(depth=10)
        self.move_subscription = self.create_subscription(
            JointTrajectory, "RobotMove", self.robot_move_callback, qos
        )
        self.gripper_subscription = self.create_subscription(
            String, "GripperControl", self.gripper_callback, 10
        )

        # 子系统
        self.tcp_server = RobotTcpServer(server_host, server_port, allowed_client, self.get_logger())
        self.gripper = GripperController(gripper_port, self.get_logger())

        self.tcp_server.start()
        if auto_init:
            self.gripper.initialize()

        # 定时重连夹爪
        self._gripper_retry_timer = self.create_timer(5.0, self._retry_gripper)

    def _retry_gripper(self):
        if not self.gripper._initialized:
            self.get_logger().info("尝试重新初始化夹爪...")
            self.gripper.initialize()

    def robot_move_callback(self, msg: JointTrajectory):
        """处理 RobotMove，向下位机转发最后一个轨迹点的 6D 位姿。"""
        try:
            if not msg.points:
                self.get_logger().warn("收到空的 JointTrajectory")
                return

            target = msg.points[-1]
            positions = [float(pos) for pos in target.positions]
            if len(positions) < 6:
                positions = positions + [0.0] * (6 - len(positions))

            position_str = ",".join(f"{pos:.6f}" for pos in positions[:6])
            send_data = f"[{position_str}]".encode("ascii")
            self.tcp_server.send_to_all(send_data)

        except Exception as e:
            self.get_logger().error(f"处理 RobotMove 消息出错：{e}")

    def gripper_callback(self, msg: String):
        command = msg.data
        if command == "open":
            self.gripper.open()
        elif command == "close":
            self.gripper.close()
        else:
            self.get_logger().warn(f"未知的夹爪控制指令: {command}")

    def run(self):
        """主循环：处理 TCP 事件和 ROS 事件。"""
        try:
            while rclpy.ok():
                lines = self.tcp_server.tick()
                for line in lines:
                    self.process_position_data(line)
                rclpy.spin_once(self, timeout_sec=0.01)
        except KeyboardInterrupt:
            pass

    def process_position_data(self, data_string: str):
        """解析下位机发送的位置数据并发布 RobotInfo。"""
        try:
            prefix = "get #real#6#"
            if data_string.startswith(prefix):
                data_string = data_string[len(prefix):].strip()

            parts = [x.strip() for x in data_string.split(",") if is_float(x.strip())]
            positions = [float(p) for p in parts]

            if len(positions) >= 6:
                robot_info = RobotInfo()
                robot_info.joint_positions = positions[:6]
                robot_info.end_positions = positions[:6]
                robot_info.state = "Normal"
                self.publisher.publish(robot_info)
                self.get_logger().info(f"发布 6 位位姿：{positions[:6]}")
            elif len(positions) > 0:
                self.get_logger().warning(f"数据不足 6 位：{positions}")

        except ValueError as e:
            self.get_logger().error(f"解析失败：{e}（原始数据：{data_string}）")

    

    def destroy_node(self):
        self.tcp_server.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IONode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
