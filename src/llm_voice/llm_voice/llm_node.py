#!/usr/bin/env python3
import os
import yaml
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from llm_voice.llm_module import LLMProcessor
from llm_voice.srv import AskText


def get_workspace_root() -> str:
    """返回仓库根目录，优先读取 ROBOARM_WS 环境变量。"""
    return os.environ.get("ROBOARM_WS", "/home/zyh/ZYH_WS")


DEFAULT_API_KEYS_PATH = f"{get_workspace_root()}/config/api_keys.yaml"


def load_zhipu_api_key(path: str) -> str:
    """从 YAML 配置文件读取智谱 API key。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 API keys 配置文件：{path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    key = data.get("zhipu_api_key", "")
    if not key or key.strip() == "YOUR_ZHIPU_API_KEY_HERE":
        raise ValueError(f"{path} 中 zhipu_api_key 未配置或仍是占位符")

    return key.strip()


class LLMNode(Node):
    def __init__(self):
        super().__init__("llm_node")

        # --------------- ROS 2 参数 ---------------
        self.declare_parameter("api_key_path", DEFAULT_API_KEYS_PATH)
        self.declare_parameter("log_path", "./log/llm.log")
        self.declare_parameter("corpus_dir", "./corpus")
        self.declare_parameter("recorder", "./tmp/recorder.wav")
        self.declare_parameter("request_timeout", 60.0)

        api_key_path = self.get_parameter("api_key_path").value
        try:
            api_key = load_zhipu_api_key(api_key_path)
        except Exception as e:
            self.get_logger().error(f"加载 API key 失败：{e}")
            raise

        # --------------- 底层模块 ---------------
        self.llm = LLMProcessor(
            api_key=api_key,
            log_path=self.get_parameter("log_path").value,
            corpus_dir=self.get_parameter("corpus_dir").value,
            recorder_file=self.get_parameter("recorder").value,
            request_timeout=self.get_parameter("request_timeout").value,
        )
        self.get_logger().info("LLM 后端预热完成")

        # --------------- service ---------------
        self.srv_txt = self.create_service(AskText, "/llm/ask_text", self.cb_ask_text)
        self.srv_aud = self.create_service(Trigger, "/llm/ask_audio", self.cb_ask_audio)

    # ---------- 回调 ----------
    def cb_ask_text(self, req, rsp):
        rsp.answer = self.llm.process_text(req.question)
        return rsp

    def cb_ask_audio(self, req, rsp):
        rsp.message = self.llm.process_audio(duration=5)
        rsp.success = True
        return rsp

    def destroy_node(self):
        self.llm.shutdown(wait=True, timeout=5.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = LLMNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
