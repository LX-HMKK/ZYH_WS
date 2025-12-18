#!/bin/bash

# 定义在新终端启动节点的函数
# 参数1: 节点启动命令
start_node() {
    local node_command=$1
    # 打开新终端，依次执行：加载bash配置、激活conda环境、加载ROS环境、启动节点
    # 最后执行bash保持终端不关闭
    gnome-terminal -- bash -c "
        source ~/.bashrc;  # 加载系统环境配置（确保conda可用）
        # conda activate grasp;  # 激活项目conda环境
        source install/setup.bash;  # 加载ROS2安装环境
        $node_command;  # 执行节点启动命令
        exec bash;  # 节点退出后保持终端打开
    "
}

# 分别启动三个节点（每个节点一个终端）
start_node "ros2 run grasp_publisher grasp_node"
start_node "ros2 run codroid_node codroid_io"
start_node "ros2 run codroid_node codroid_move_test"

echo "三个节点已分别在新终端启动"