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

# 节点：
## grasp_node：

## codroid_io：
发送至下位机需要广播，下位机的端口是随机的



# 假grasp_node测试
只一次
```bash
ros2 topic pub /grasp_result grasp_interfaces/msg/GraspResult "{
  trans_cam: [0.0,0.0,0.0],
  rot_cam_flat: [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
  width: 0.05,
  score: 0.95,
  pos_base: [-25.928, -427.63, -27.168],
  euler_base: [21.394, 40.531, 10.292],
  cls_name: 'banana',
  stamp: {sec: 0, nanosec: 0}
}" --once
```
多次，[-r 1]是1Hz
```bash
ros2 topic pub /grasp_result grasp_interfaces/msg/GraspResult "{
  trans_cam: [0.0,0.0,0.0],
  rot_cam_flat: [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],
  width: 0.05,
  score: 0.95,
  pos_base: [-25.928, -427.63, -27.168],
  euler_base: [21.394, 40.531, 10.292],
  cls_name: 'banana',
  stamp: {sec: 0, nanosec: 0}
}" -r 1
```

[RobotCmd] 机器人未处于自动模式-空闲状态, 拒绝响应运动指令.