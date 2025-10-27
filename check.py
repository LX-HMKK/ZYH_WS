import os
from collections import defaultdict

# --- 配置 ---
CLASSES_FILE = 'D:/StudyWorks/3.1/ZYH_ws/images/classes.txt'      # 类别文件路径
LABELS_DIR = 'D:/StudyWorks/3.1/ZYH_ws/labels'     # 标签文件夹路径
# ------------

def load_class_names(classes_file):
    """从 classes.txt 文件加载类别名称。"""
    try:
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
        return class_names
    except FileNotFoundError:
        print(f"错误：找不到类别文件 {classes_file}。请检查路径。")
        return None
    except Exception as e:
        print(f"加载类别文件时出错：{e}")
        return None

def analyze_labels(labels_dir, class_names):
    """
    遍历标签文件，统计类别次数，并检查同一文件中同一类别是否多次出现。
    """
    if not class_names:
        return

    # 初始化总计数器：{class_id: count}
    total_class_counts = defaultdict(int)
    # 初始化警告列表
    warnings = []
    
    # 获取标签文件夹中所有的 .txt 文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    print(f"找到 {len(label_files)} 个标签文件。")

    for filename in label_files:
        filepath = os.path.join(labels_dir, filename)
        
        # 记录当前文件中每个类别的出现次数：{class_id: count}
        file_class_counts = defaultdict(int)
        
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # YOLO 标签文件每行第一个是类别 ID
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        print(f"警告：文件 {filename} 第 {line_num} 行的类别ID不是整数，已跳过。")
                        continue

                    # 检查 class_id 是否在有效范围内
                    if 0 <= class_id < len(class_names):
                        file_class_counts[class_id] += 1
                        total_class_counts[class_id] += 1
                    else:
                        warnings.append(f"警告：文件 {filename} 第 {line_num} 行的类别ID {class_id} 超出范围 (0-{len(class_names)-1})。")

        except Exception as e:
            warnings.append(f"错误：读取文件 {filename} 时出错：{e}")
            continue

        # 检查当前文件中的警告：同一类别出现多次
        for class_id, count in file_class_counts.items():
            if count > 1:
                class_name = class_names[class_id] if class_id < len(class_names) else f"ID_{class_id}"
                warnings.append(
                    f"警告：文件 {filename} 中类别 '{class_name}' (ID: {class_id}) 出现了 {count} 次。"
                )

    # --- 结果输出 ---
    print("\n--- 类别标记总次数 ---")
    
    # 将结果转换为列表，并根据类别ID排序，方便查看
    sorted_counts = sorted(total_class_counts.items(), key=lambda item: item[0])

    for class_id, count in sorted_counts:
        class_name = class_names[class_id] if class_id < len(class_names) else f"未知类_{class_id}"
        print(f"ID: {class_id} | 类别 '{class_name}': {count} 次")
        
    # 打印警告
    if warnings:
        print("\n--- 警告信息 ---")
        for warning in warnings:
            print(warning)
    else:
        print("\n--- 警告信息 ---")
        print("未发现任何警告。")

if __name__ == "__main__":
    # 1. 加载类别名称
    class_names = load_class_names(CLASSES_FILE)

    if class_names:
        # 2. 检查标签文件夹是否存在
        if not os.path.isdir(LABELS_DIR):
            print(f"错误：找不到数据集标签文件夹 {LABELS_DIR}。请检查配置。")
        else:
            # 3. 分析标签文件
            analyze_labels(LABELS_DIR, class_names)