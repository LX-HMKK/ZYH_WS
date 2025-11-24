import numpy as np
import pyrealsense2 as rs
import cv2

class Camera(object):
    def __init__(self, width=1280, height=720, fps=30):
        self.im_height = height
        self.im_width = width
        self.fps = fps
        self.intrinsics = None
        self.scale = None
        self.pipeline = rs.pipeline()  # 初始化 pipeline
        self.connect()  # 连接摄像头并配置参数

    def connect(self):
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        # 启动视频流
        cfg = self.pipeline.start(config)

        # 获取 RGB 相机内参
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = self.get_intrinsics(rgb_profile)

        # 获取深度尺度（将原始数据转换为米）
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        print("深度尺度:", self.scale)
        print("D435 已连接...")

    def get_data(self):
        # 等待对齐后的帧（深度图对齐到彩色图）
        frames = self.pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        # 获取对齐后的深度图和彩色图
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 转换为 NumPy 数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale  # 转换为米（可选）
        depth_image = np.expand_dims(depth_image, axis=2)  # 添加通道维度 (H,W,1)

        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def plot_image(self):
        color_image, depth_image = self.get_data()

        # 将深度图转换为伪彩色（8-bit）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image[:, :, 0], alpha=0.03),  # 注意取单通道
            cv2.COLORMAP_JET
        )

        # 调整彩色图尺寸以匹配深度图（如果分辨率不同）
        if depth_colormap.shape != color_image.shape:
            resized_color_image = cv2.resize(
                color_image,
                dsize=(depth_colormap.shape[1], depth_colormap.shape[0]),
                interpolation=cv2.INTER_AREA
            )
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # 显示图像
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(5000)  # 显示 5 秒
        cv2.destroyAllWindows()

    def get_intrinsics(self, rgb_profile):
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        print("相机内参:", raw_intrinsics)
        # 转换为 3x3 矩阵形式
        intrinsics = np.array([
            [raw_intrinsics.fx, 0, raw_intrinsics.ppx],
            [0, raw_intrinsics.fy, raw_intrinsics.ppy],
            [0, 0, 1]
        ])
        return intrinsics

    def release(self):
        self.pipeline.stop()  # 释放资源

if __name__ == '__main__':
    mycamera = Camera()  # 自动调用 __init__ -> connect()
    try:
        # 显示图像
        mycamera.plot_image()
        
        # 如果需要获取内参矩阵，可以这样调用（但通常不需要手动调用）
        # intrinsics = mycamera.intrinsics
        # print("内参矩阵:\n", intrinsics)
        
    finally:
        mycamera.release()  # 确保资源释放