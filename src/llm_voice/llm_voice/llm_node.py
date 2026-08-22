#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from llm_voice.llm_module import LLMProcessor
from arm_interfaces.srv import AskText
from arm_utils import get_workspace_root
from llm_voice.api_key_utils import load_zhipu_api_key


DEFAULT_API_KEYS_PATH = f"{get_workspace_root()}/config/api_keys.yaml"
DEFAULT_CORPUS_DIR = f"{get_workspace_root()}/src/llm_voice/corpus"
DEFAULT_LOG_PATH = f"{get_workspace_root()}/log/llm.log"
DEFAULT_RECORDER = f"{get_workspace_root()}/tmp/recorder.wav"


class LLMNode(Node):
    def __init__(self):
        super().__init__("llm_node")

        # --------------- ROS 2 参数 ---------------
        self.declare_parameter("api_key_path", DEFAULT_API_KEYS_PATH)
        self.declare_parameter("log_path", DEFAULT_LOG_PATH)
        self.declare_parameter("corpus_dir", DEFAULT_CORPUS_DIR)
        self.declare_parameter("recorder", DEFAULT_RECORDER)
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
