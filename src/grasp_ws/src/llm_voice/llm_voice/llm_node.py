#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from llm_voice.llm_module import LLMProcessor      # ← 改这里
from llm_voice.srv import AskText                  # ← 改这里

class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')

        # --------------- ROS 2 参数 ---------------
        self.declare_parameter('api_key',   os.getenv('ZHIPU_API_KEY', ''))
        self.declare_parameter('log_path',  './log/llm.log')
        self.declare_parameter('corpus_dir','./corpus')
        self.declare_parameter('recorder' , './tmp/recorder.wav')

        api_key   = self.get_parameter('api_key').value
        if not api_key:
            self.get_logger().error('ZHIPU_API_KEY 未设置！')
            quit()

        # --------------- 底层模块 ---------------
        self.llm = LLMProcessor(
            api_key     = api_key,
            log_path    = self.get_parameter('log_path').value,
            corpus_dir  = self.get_parameter('corpus_dir').value,
            recorder_file=self.get_parameter('recorder').value
        )
        self.get_logger().info('LLM 后端预热完成')

        # --------------- service ---------------
        self.srv_txt = self.create_service(AskText,  '/llm/ask_text',  self.cb_ask_text)
        self.srv_aud = self.create_service(Trigger,  '/llm/ask_audio',  self.cb_ask_audio)

    # ---------- 回调 ----------
    def cb_ask_text(self, req, rsp):
        rsp.answer = self.llm.process_text(req.question)
        return rsp

    def cb_ask_audio(self, req, rsp):
        rsp.message = self.llm.process_audio(duration=5)
        rsp.success = True
        return rsp


def main():
    rclpy.init()
    node = LLMNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
