# 编译
```bash
cd ~/zyh_demo1/graspnet-baseline-main/wsq_ws

# 1. 清理旧编译缓存（防止残留）
rm -rf build/grasp_publisher install/grasp_publisher

# 2. 重新编译（一定带 --symlink-install）
python -m colcon build --symlink-install

# 3. 重新 source
source install/setup.bash

# 4. 验证接口已生成
ros2 interface show grasp_publisher/msg/GraspResult
# 若能正常打印消息内容，说明生成成功
```

# 运行
```bash
ros2 run grasp_publisher grasp_node
```

# 文件结构
src/grap_publisher/grap_publisher/*：节点代码，引用/kw/robot.py中的函数    
src/grap_publisher/msg/GrapResult.msg：节点信息定义