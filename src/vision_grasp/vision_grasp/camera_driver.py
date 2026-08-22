#coding=utf-8
"""Intel RealSense 相机封装，统一初始化、重启、帧获取与资源释放。"""
import time
import logging
from typing import Tuple, Optional

import pyrealsense2 as rs

from grasp_pipeline import Config


class _StdoutLogger:
    """兼容 ROS logger 的默认日志输出（当未传入 logger 时使用）。"""

    def info(self, msg: str):
        print(f"[INFO] {msg}")

    def warn(self, msg: str):
        print(f"[WARN] {msg}")

    def error(self, msg: str):
        print(f"[ERROR] {msg}")


class RealSenseCamera:
    """RealSense D435/D435i 相机驱动封装。

    通过注入的 Config 实例读取分辨率、帧率、对齐方式等参数，
    并提供统一的启动、重启、获取对齐帧、停止接口。
    """

    def __init__(
        self,
        config: Config,
        logger=None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        startup_delay: float = 4.0,
        color_format=rs.format.rgb8,
    ):
        """
        Args:
            config: 配置对象，需包含 CAMERA_RES、CAMERA_FPS、ALIGN_WAY、USE_ROS_BAG、BAG_PATH。
            logger: 日志对象，需有 info/warn/error 方法。默认打印到 stdout。
            max_retries: 启动失败最大重试次数。
            retry_delay: 每次重试间隔（秒）。
            startup_delay: 启动成功后等待固件 ready 的时间（秒）。
            color_format: RealSense 彩色流格式，默认 rs.format.rgb8；eyeInHand 等场景可传 rs.format.bgr8。
        """
        if config is None:
            raise ValueError("RealSenseCamera 必须注入 Config 实例")
        self.config = config
        self.logger = logger if logger is not None else _StdoutLogger()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.startup_delay = startup_delay
        self.color_format = color_format

        self.pipeline: Optional[rs.pipeline] = None
        self.aligner: Optional[rs.align] = None
        self.depth_scale: Optional[float] = None

    def start(self) -> Tuple[rs.pipeline, rs.align, float]:
        """启动 RealSense pipeline，返回 (pipeline, aligner, depth_scale)。

        失败时会按 max_retries 重试，全部失败后抛出最后一次异常。
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                pipeline = rs.pipeline()
                cfg = rs.config()

                if self.config.USE_ROS_BAG:
                    cfg.enable_device_from_file(self.config.BAG_PATH)
                else:
                    cfg.enable_stream(
                        rs.stream.color,
                        self.config.CAMERA_RES[0],
                        self.config.CAMERA_RES[1],
                        self.color_format,
                        self.config.CAMERA_FPS,
                    )
                    cfg.enable_stream(
                        rs.stream.depth,
                        self.config.CAMERA_RES[0],
                        self.config.CAMERA_RES[1],
                        rs.format.z16,
                        self.config.CAMERA_FPS,
                    )

                aligner = rs.align(rs.stream.color if self.config.ALIGN_WAY else rs.stream.depth)
                profile = pipeline.start(cfg)
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

                self.logger.info(f"深度比例系数：{depth_scale:.6f} 米/像素")
                time.sleep(self.startup_delay)
                self.logger.info(f"RealSense camera initialized successfully on attempt {attempt}")

                self.pipeline = pipeline
                self.aligner = aligner
                self.depth_scale = depth_scale
                return pipeline, aligner, depth_scale

            except Exception as e:
                last_error = e
                self.logger.warn(f"Attempt {attempt} failed to initialize RealSense camera: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        self.logger.error("Failed to initialize RealSense camera after all retries")
        raise last_error

    def stop(self):
        """停止 pipeline 并清理资源。"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                self.logger.info("RealSense pipeline stopped")
            except Exception as e:
                self.logger.warn(f"停止 RealSense pipeline 时出错: {e}")
            finally:
                self.pipeline = None
                self.aligner = None
                self.depth_scale = None

    def restart(self) -> Tuple[rs.pipeline, rs.align, float]:
        """先停止再重新启动相机，返回 (pipeline, aligner, depth_scale)。"""
        self.logger.info("正在重启相机...")
        self.stop()
        time.sleep(1.0)   # 等硬件掉线
        result = self.start()
        time.sleep(2.0)   # 等硬件重新稳定
        self.logger.info("相机重启成功")
        return result

    def get_aligned_frames(self, timeout_ms: int = 2000) -> rs.composite_frame:
        """等待并返回对齐后的帧。"""
        if self.pipeline is None:
            raise RuntimeError("RealSense pipeline 未启动，请先调用 start()")
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        return self.aligner.process(frames)

    def is_running(self) -> bool:
        """返回相机是否已启动。"""
        return self.pipeline is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
