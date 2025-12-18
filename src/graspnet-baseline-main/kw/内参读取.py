import pyrealsense2 as rs

# 初始化管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 深度流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 彩色流

# 启动设备
profile = pipeline.start(config)

# 获取深度流内参
depth_profile = profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
print("深度相机内参矩阵 (K_depth):")
print(f"fx = {depth_intrinsics.fx:.6f}, fy = {depth_intrinsics.fy:.6f}")
print(f"cx = {depth_intrinsics.ppx:.6f}, cy = {depth_intrinsics.ppy:.6f}")

# 获取彩色流内参
color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
print("\n彩色相机内参矩阵 (K_color):")
print(f"fx = {color_intrinsics.fx:.6f}, fy = {color_intrinsics.fy:.6f}")
print(f"cx = {color_intrinsics.ppx:.6f}, cy = {color_intrinsics.ppy:.6f}")
print(f"width = {color_intrinsics.width:.6f}, height  = {color_intrinsics.height :.6f}")

# 停止管道
pipeline.stop()
