#coding=utf-8
"""RealSense 帧对齐与预处理。"""
import numpy as np
import cv2


class FrameProcessor:
    """处理 RealSense 对齐/未对齐帧，转换为模型可用的图像。"""

    @staticmethod
    def process_aligned_frames(frames, aligner, use_bag: bool):
        """
        处理对齐的彩色图与深度图。

        Returns:
            color_img (RGB), depth_img (16位), depth_colormap (用于显示)
        """
        aligned_frames = aligner.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 转换彩色图（BGR→RGB，适配后续模型）
        color_img = np.asanyarray(color_frame.get_data())
        if use_bag:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # 转换深度图（16位格式，保留原始深度）
        depth_img = np.asanyarray(depth_frame.get_data())

        # 生成深度彩色映射图（用于显示）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.05),
            cv2.COLORMAP_JET
        )

        return color_img, depth_img, depth_colormap

    @staticmethod
    def process_unaligned_frames(frames, use_bag: bool, align_way: int):
        """
        处理未对齐的原始帧（含红外图显示）。

        Returns:
            color_img, depth_img
        """
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        # 适配 ROS Bag 尺寸不一致问题
        if use_bag:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            if align_way:
                depth_img = cv2.resize(depth_img, (color_img.shape[1], color_img.shape[0]))
            else:
                color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]))
        else:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            if align_way:
                depth_img = cv2.resize(depth_img, (color_img.shape[1], color_img.shape[0]))
            else:
                color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]))

        return color_img, depth_img


# 保持与原函数签名兼容的模块级函数
process_aligned_frames = FrameProcessor.process_aligned_frames
process_unaligned_frames = FrameProcessor.process_unaligned_frames
