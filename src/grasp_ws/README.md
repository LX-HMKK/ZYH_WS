# 编译

```bash
conda activate grasp
python -m colcon build --symlink-install
```

# 运行
```bash
source install/setup.bash
ros2 run grasp_publisher grasp_node
ros2 run codroid_node codroid_io
ros2 run codroid_node codroid_move_test
```
[RobotCmd] 机器人未处于自动模式-空闲状态, 拒绝响应运动指令.