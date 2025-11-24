std_msgs/Header header

float64[] joint_positions # 各关节位置(rad)

float64[] end_positions # 末端位置(m, rad)

string state # 机器人状态
             # "PowerOff": 下电状态
             # "Idle": 已上电，且处于手动模式-空闲
             # "AutoReady": 已上电，且处于自动模式-空闲
             # "AutoRunning": 已上电，且处于自动模式-运动中
             # 其他未提及状态几乎不会使用

int8 fault_flag # 系统错误标志
                # 该标志为0时，表示系统当前不存在错误
                # 否则表示系统当前存在错误，需要手动清除错误方可继续使用