#!/usr/bin/env python3
"""
RealSense相机连接测试脚本
"""

import pyrealsense2 as rs
import time


def test_camera_connection():
    print("开始测试RealSense相机连接...")
    
    # 创建上下文并查询设备
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"发现 {len(devices)} 个连接的设备")
    
    if len(devices) == 0:
        print("错误: 未发现连接的RealSense设备")
        return False
    
    # 打印设备信息
    for i, device in enumerate(devices):
        print(f"\n设备 {i}:")
        try:
            name = device.get_info(rs.camera_info.name)
            print(f"  名称: {name}")
        except:
            pass
            
        try:
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"  序列号: {serial}")
        except:
            pass
            
        try:
            fw_version = device.get_info(rs.camera_info.firmware_version)
            print(f"  固件版本: {fw_version}")
        except:
            pass
    
    # 尝试启动相机管道
    print("\n尝试启动相机管道...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 尝试标准配置 (1280x720)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        profile = pipeline.start(config)
        print("成功启动相机管道 (1280x720)")
        
        # 尝试获取几帧数据
        for i in range(5):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                print(f"成功获取第 {i+1} 帧数据")
            else:
                print(f"第 {i+1} 帧数据不完整")
        
        pipeline.stop()
        print("测试完成，相机工作正常")
        return True
        
    except Exception as e:
        print(f"启动相机管道失败: {str(e)}")
        
        # 尝试更低的分辨率
        print("\n尝试使用较低分辨率 (640x480)...")
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            profile = pipeline.start(config)
            print("成功启动相机管道 (640x480)")
            
            # 尝试获取几帧数据
            for i in range(5):
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    print(f"成功获取第 {i+1} 帧数据 (低分辨率)")
                else:
                    print(f"第 {i+1} 帧数据不完整")
            
            pipeline.stop()
            print("低分辨率测试完成，相机工作正常")
            return True
            
        except Exception as e2:
            print(f"低分辨率测试也失败了: {str(e2)}")
            return False



def init_realsense(self):
    import pyrealsense2 as rs
    # 添加重试机制和延时，让初始化更稳定
    max_retries = 5
    retry_delay = 1.0  # 延迟时间（秒）
    
    # 首先检查是否有连接的设备
    ctx = rs.context()
    devices = ctx.query_devices()
    self.get_logger().info(f'发现 {len(devices)} 个连接的设备')
    
    for i, device in enumerate(devices):
        self.get_logger().info(f'设备 {i}: {device.get_info(rs.camera_info.name)}')
        self.get_logger().info(f'  序列号: {device.get_info(rs.camera_info.serial_number)}')
        self.get_logger().info(f'  固件版本: {device.get_info(rs.camera_info.firmware_version)}')
    
    if len(devices) == 0:
        self.get_logger().error('未发现连接的RealSense设备')
        raise RuntimeError("未发现连接的RealSense设备")
    
    for attempt in range(max_retries):
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, *Config.CAMERA_RES, rs.format.rgb8, Config.CAMERA_FPS)
            cfg.enable_stream(rs.stream.depth, *Config.CAMERA_RES, rs.format.z16, Config.CAMERA_FPS)

            align_to = rs.stream.color
            aligner = rs.align(align_to)

            profile = pipeline.start(cfg)
            
            # 获取深度传感器和深度比例
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            self.get_logger().info(f'深度比例系数：{depth_scale:.6f} 米/像素')
            
            # 添加额外的等待时间，确保相机完全初始化
            time.sleep(3.0)
            
            self.get_logger().info(f'RealSense camera initialized successfully on attempt {attempt + 1}')
            return pipeline, aligner, depth_scale
            
        except Exception as e:
            self.get_logger().warn(f'Attempt {attempt + 1} failed to initialize RealSense camera: {str(e)}')
            if attempt < max_retries - 1:
                self.get_logger().info(f'Waiting {retry_delay} seconds before retry...')
                time.sleep(retry_delay)
            else:
                self.get_logger().error('Failed to initialize RealSense camera after all retries')
                raise e
                
def restart_camera(self):
    """重启相机连接"""
    try:
        self.get_logger().info("正在重启相机...")
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
            
        # 等待一段时间
        time.sleep(2.0)
        
        # 重新初始化相机
        self.pipeline, self.aligner, self.depth_scale = self.init_realsense()
        self.get_logger().info("相机重启成功")
    except Exception as e:
        self.get_logger().error(f"相机重启失败: {str(e)}")
        # 如果重启失败，尝试使用更低的分辨率
        self.try_lower_resolution()
        
def try_lower_resolution(self):
    """尝试使用较低的分辨率初始化相机"""
    import pyrealsense2 as rs
    self.get_logger().info("尝试使用较低的分辨率初始化相机...")
    
    # 尝试使用640x480分辨率
    lower_res = (640, 480)
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, *lower_res, rs.format.rgb8, Config.CAMERA_FPS)
        cfg.enable_stream(rs.stream.depth, *lower_res, rs.format.z16, Config.CAMERA_FPS)
        
        align_to = rs.stream.color
        aligner = rs.align(align_to)
        
        profile = pipeline.start(cfg)
        
        # 更新配置中的分辨率
        Config.CAMERA_RES = lower_res
        
        # 获取深度传感器和深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        self.get_logger().info(f'使用较低分辨率 {lower_res} 初始化成功，深度比例系数：{depth_scale:.6f} 米/像素')
        
        self.pipeline, self.aligner, self.depth_scale = pipeline, aligner, depth_scale
        return
    except Exception as e:
        self.get_logger().error(f"使用较低分辨率初始化也失败了: {str(e)}")
        raise RuntimeError("无法初始化RealSense相机，即使降低了分辨率")


if __name__ == "__main__":
    test_camera_connection()