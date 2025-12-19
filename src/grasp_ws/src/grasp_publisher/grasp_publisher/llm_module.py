# llm_module.py
import os
import time
import json
import logging
import subprocess  # 替代 os.system，方便捕获输出
from threading import Thread, Event
from zhipuai import ZhipuAI



PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class LLMProcessor:
    """
    兼容 Linux 的 LLM 处理器，Windows 下调试时录音功能自动降级。
    """

    def __init__(self,
                 api_key: str,
                 log_path: str = "./log/llm.log",
                 corpus_dir: str = "./corpus",
                 recorder_file: str = "./tmp/recorder.wav"):
        # 1. 基础路径 & 文件\
   
        self.recorder_file = recorder_file
        self.log_path = log_path
        self.corpus_dir = os.path.join(PACKAGE_DIR, "corpus")   # 与 llm_module.py 同级的 corpus 文件夹
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.recorder_file), exist_ok=True)

        # 2. 日志
        self._init_logger()

        # 3. 智谱客户端
        self.client = ZhipuAI(api_key=api_key.strip())
        self.messages = []  # 与旧版保持一致
        self.last_response = ""

        # 4. 麦克风参数（Linux下）
        self.microphone_card = "0"
        self.microphone_device = "0"

        # 5. 后台线程
        self.event = Event()
        self.llm_thread = Thread(target=self._llm_worker, daemon=True)
        self.llm_thread.start()
        self.logger.info("LLMProcessorLite 初始化完成")

    # ---------------- 对外接口 ----------------
    def process_text(self, text: str) -> str:
        """纯文本→LLM，阻塞返回回答"""
        self.logger.info(f"[TEXT] 输入：{text}")
        self._push_job(text, source="text")
        return self.last_response

    def process_audio(self, duration: int = 5) -> str:
        """录音→ASR→LLM，阻塞返回回答"""
        self.logger.info(f"[AUDIO] 请求录音 {duration}s")
        self._push_job("", source="audio", duration=duration)
        return self.last_response

    # ---------------- 内部实现 ----------------
    def _init_logger(self):
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _push_job(self, text: str, source: str, duration: int = 5):
        """把任务丢给后台线程，主线程阻塞等待 event"""
        self.job = {"text": text, "source": source, "duration": duration}
        self.event.clear()
        self.event.wait()  # 后台完成后会 set()

    def _llm_worker(self):
        """后台真正干活的线程"""
        try:
            # 1. 预热语料（加载 corpus1.txt/2/3.txt），但不污染 messages
            self._warm_up()
            while True:
                if hasattr(self, "job"):
                    job = self.job
                    del self.job  # 一次消费
                    self._handle_job(job)
                time.sleep(0.1)
        except Exception as e:
            self.logger.exception("后台线程崩溃")

    def _warm_up(self):
        """加载 corpus1.txt.txt ~ corpus3.txt 作为系统提示"""
        for i in range(1, 4):
            file_i = os.path.join(self.corpus_dir, f"corpus{i}.txt")
            if not os.path.isfile(file_i):
                self.logger.warning(f"语料文件不存在：{file_i}，跳过预热")
                continue
            with open(file_i, encoding="utf-8") as f:
                content = f.read().strip()
            self.messages.append({"role": "user", "content": content})
            answer, _ = self._chat_with_glm(self.messages)
            self.messages.append({"role": "assistant", "content": answer})
        self.logger.info("语料预热完成")

    def _handle_job(self, job: dict):
        """处理单次任务"""
        try:
            if job["source"] == "text":
                text = job["text"]
            else:  # audio
                text = self._record_and_asr(job["duration"])
                if text.startswith("【录音失败】") or text.startswith("【识别失败】"):
                    self.last_response = text
                    self.event.set()
                    return

            # 与大模型对话
            self.messages.append({"role": "user", "content": text})
            answer, _ = self._chat_with_glm(self.messages)
            self.messages.append({"role": "assistant", "content": answer})
            self.last_response = answer
            self.logger.info(f"[GLM] 回答：{answer}")
        except Exception as e:
            self.logger.exception("任务处理异常")
            self.last_response = f"【LLM 异常】{e}"
        finally:
            self.event.set()

    # ---------------- 录音+ASR（Linux only） ----------------
    def _record_and_asr(self, duration: int) -> str:
        """返回识别文本；Windows 下直接返回失败提示"""
        if os.name == "nt":  # Windows
            self.logger.info("Windows 环境，录音功能暂不可用")
            return "【录音失败】当前为 Windows 调试模式"

        # 1. 录音
        cmd = f"arecord -D hw:{self.microphone_card},{self.microphone_device} -f cd -d {duration} -c 1 {self.recorder_file}"
        ret = subprocess.run(cmd, shell=True, capture_output=True)
        if ret.returncode != 0 or not os.path.getsize(self.recorder_file):
            self.logger.error("录音失败")
            return "【录音失败】"

        # 2. 语音识别
        try:
            with open(self.recorder_file, "rb") as f:
                resp = self.client.audio.transcriptions.create(
                    model="glm-asr", file=f, stream=False
                )
            text = resp.text.strip()
            self.logger.info(f"[ASR] 识别结果：{text}")
            return text
        except Exception as e:
            self.logger.exception("ASR 异常")
            return "【识别失败】"

    # ---------------- 大模型调用 ----------------
    def _chat_with_glm(self, messages: list) -> tuple:
        st = time.time()
        try:
            rsp = self.client.chat.completions.create(
                model="GLM-4-Flash-250414",
                messages=messages
            )
            answer = rsp.choices[0].message.content.strip()
            cost = time.time() - st
            self.logger.debug(f"GLM 调用耗时：{cost:.2f}s")
            return answer, cost
        except Exception as e:
            self.logger.exception("GLM 调用失败")
            return f"【GLM 异常】{e}", time.time() - st


if __name__ == "__main__":
    api_key = os.getenv("ZHIPU_API_KEY", "e95feaa7f6ab4cdc807038f0b823a952.twjJFz6nrRg1GylJ")
    proc = LLMProcessor(api_key=api_key)
    time.sleep(1)  # 等后台预热完

    ## 测试1：文本处理
    #print("=== 文本测试 ===")
    #ans = proc.process_text("我需要一只马克笔")
    #print("回答：", ans)

    #测试2：录音处理
    print("=== 录音测试 ===")
    print("请说话：(5s录音)")
    ans = proc.process_audio(duration=5)
    print("回答：", ans)