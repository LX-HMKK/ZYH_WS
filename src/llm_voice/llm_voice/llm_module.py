# llm_module.py
import os
import time
import logging
import subprocess
from typing import Optional
from queue import Queue, Empty
from threading import Thread
from concurrent.futures import Future
from zhipuai import ZhipuAI


class LLMProcessor:
    """
    线程安全的 LLM 处理器。

    每个 process_text / process_audio 调用会生成一个独立任务，
    通过 Future 等待结果，避免多线程共享 event/job/last_response 导致的问题。
    """

    def __init__(
        self,
        api_key: str,
        log_path: str = "./log/llm.log",
        corpus_dir: str = "./corpus",
        recorder_file: str = "./tmp/recorder.wav",
        request_timeout: float = 60.0,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("api_key 不能为空")

        self.recorder_file = recorder_file
        self.log_path = log_path
        self.corpus_dir = corpus_dir
        self.request_timeout = request_timeout

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.recorder_file) or ".", exist_ok=True)

        self._init_logger()
        self.client = ZhipuAI(api_key=api_key.strip())
        self.messages: list[dict] = []

        self.microphone_card = "0"
        self.microphone_device = "0"

        self._job_queue: Queue[dict] = Queue()
        self._worker = Thread(target=self._llm_worker, daemon=True)
        self._worker.start()
        self.logger.info("LLMProcessor 初始化完成")

    # ---------------- 对外接口 ----------------
    def process_text(self, text: str) -> str:
        """纯文本→LLM，阻塞返回回答。"""
        self.logger.info(f"[TEXT] 输入：{text}")
        return self._submit_job(text, source="text")

    def process_audio(self, duration: int = 5) -> str:
        """录音→ASR→LLM，阻塞返回回答。"""
        self.logger.info(f"[AUDIO] 请求录音 {duration}s")
        return self._submit_job("", source="audio", duration=duration)

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """发送关闭信号并等待工作线程结束。"""
        self._job_queue.put({"_shutdown": True})
        if wait:
            self._worker.join(timeout=timeout)

    # ---------------- 内部实现 ----------------
    def _init_logger(self):
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _submit_job(self, text: str, source: str, duration: int = 5) -> str:
        future: Future[str] = Future()
        job = {
            "text": text,
            "source": source,
            "duration": duration,
            "future": future,
        }
        self._job_queue.put(job)
        try:
            return future.result(timeout=self.request_timeout)
        except Exception as e:
            self.logger.exception("LLM 调用超时或异常")
            return f"【LLM 异常】{e}"

    def _llm_worker(self):
        """后台工作线程：预热语料并循环处理任务。"""
        try:
            self._warm_up()
        except Exception:
            self.logger.exception("语料预热失败")

        while True:
            try:
                job = self._job_queue.get(timeout=0.5)
            except Empty:
                continue

            if job.get("_shutdown"):
                break

            try:
                result = self._handle_job(job)
                job["future"].set_result(result)
            except Exception as e:
                self.logger.exception("任务处理异常")
                job["future"].set_result(f"【LLM 异常】{e}")

    def _warm_up(self):
        """加载 corpus1.txt ~ corpus3.txt 作为系统提示。"""
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

    def _handle_job(self, job: dict) -> str:
        """处理单次任务并返回回答字符串。"""
        if job["source"] == "text":
            text = job["text"]
        else:  # audio
            text = self._record_and_asr(job["duration"])
            if text.startswith("【录音失败】") or text.startswith("【识别失败】"):
                return text

        self.messages.append({"role": "user", "content": text})
        answer, _ = self._chat_with_glm(self.messages)
        self.messages.append({"role": "assistant", "content": answer})
        self.logger.info(f"[GLM] 回答：{answer}")
        return answer

    # ---------------- 录音+ASR（Linux only） ----------------
    def _record_and_asr(self, duration: int) -> str:
        """返回识别文本；Windows 下直接返回失败提示。"""
        if os.name == "nt":
            self.logger.info("Windows 环境，录音功能暂不可用")
            return "【录音失败】当前为 Windows 调试模式"

        cmd = [
            "arecord",
            "-D", f"hw:{self.microphone_card},{self.microphone_device}",
            "-f", "cd",
            "-d", str(duration),
            "-c", "1",
            self.recorder_file,
        ]
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode != 0 or not os.path.getsize(self.recorder_file):
            self.logger.error(f"录音失败：{ret.stderr.decode('utf-8', errors='ignore')}")
            return "【录音失败】"

        try:
            with open(self.recorder_file, "rb") as f:
                resp = self.client.audio.transcriptions.create(
                    model="glm-asr", file=f, stream=False
                )
            text = resp.text.strip()
            self.logger.info(f"[ASR] 识别结果：{text}")
            return text
        except Exception:
            self.logger.exception("ASR 异常")
            return "【识别失败】"

    # ---------------- 大模型调用 ----------------
    def _chat_with_glm(self, messages: list) -> tuple[str, float]:
        st = time.time()
        try:
            rsp = self.client.chat.completions.create(
                model="GLM-4-Flash-250414",
                messages=messages,
            )
            answer = rsp.choices[0].message.content.strip()
            cost = time.time() - st
            self.logger.debug(f"GLM 调用耗时：{cost:.2f}s")
            return answer, cost
        except Exception:
            self.logger.exception("GLM 调用失败")
            return f"【GLM 异常】", time.time() - st


if __name__ == "__main__":
    import sys

    api_key = os.getenv("ZHIPU_API_KEY", "")
    if not api_key:
        print("请设置环境变量 ZHIPU_API_KEY 或在 api_keys.yaml 中配置")
        sys.exit(1)

    proc = LLMProcessor(api_key=api_key)
    time.sleep(1)
    print("=== 文本测试 ===")
    ans = proc.process_text("你好")
    print("回答：", ans)
    proc.shutdown()
