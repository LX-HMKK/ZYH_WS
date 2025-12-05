# 手眼标定

## 1. 运行

```bash
python3 保存RGB图像.py  #调用深度相机保存棋盘格图片30张
```

```bash
python3 eye_in_hand.py  #执行手眼标定代码，对保存后的图像进行识别参数
```
## 2. 参数

```python
pose_vectors = np.array([])
```
每一张标定图片所对应的机械臂末端位姿 $$[x,y,z,rx,ry.rz]$$


