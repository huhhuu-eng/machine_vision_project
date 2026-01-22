import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import torch

# -------------------------- 1. 基础配置 --------------------------

IMG_PATH = "bike.jpg"

SAVE_DIR = ""

WEIGHTS_PATH = "yolov5su.pt"
# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------- 2. 权重文件检查与加载 --------------------------
def check_and_load_model(weights_path):
    """检查权重文件是否存在，不存在则手动下载"""
    try:
        # 尝试加载模型
        model = YOLO(weights_path)
        print(f"模型加载成功，权重文件路径：{weights_path}")
        return model
    except FileNotFoundError:

        print(f" {weights_path} 下载失败，尝试手动指定官方权重URL...")
        
        model = YOLO(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{weights_path}")
        return model
    except Exception as e:
        raise Exception(f"模型加载失败：{e}")


model = check_and_load_model(WEIGHTS_PATH)

# -------------------------- 3. 目标检测   --------------------------
def detect_shared_bike(img_path, save_dir):
    # 步骤1：读取图片
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片，请检查路径：{img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

    # 步骤2：特征提取+目标定位+分类判断（YOLOv5核心推理）
    # conf=0.5：置信度阈值，过滤低置信度检测结果；classes=[1]：仅检测COCO的"bicycle"类（类别ID=1）
    results = model(img, conf=0.5, classes=[1])

    # 步骤3：解析检测结果（提取共享单车位置）
    bike_boxes = []  
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bike_boxes.append([x1, y1, x2, y2])
            # 在图片上绘制边界框和标签
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框，线宽2
            cv2.putText(img, "Shared Bike", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 步骤4：保存检测结果
    # 保存带检测框的图片
    result_img_path = os.path.join(save_dir, "detected_bike.jpg")
    cv2.imwrite(result_img_path, img)
    # 保存检测位置数据
    box_txt_path = os.path.join(save_dir, "bike_position.txt")
    with open(box_txt_path, 'w', encoding='utf-8') as f:
        f.write("共享单车边界框坐标（x1, y1, x2, y2）：\n")
        for i, box in enumerate(bike_boxes):
            f.write(f"第{i+1}辆单车：{box}\n")

    # 步骤5：可视化检测结果
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("校园共享单车检测结果")
    plt.savefig(os.path.join(save_dir, "detection_visualization.png"), bbox_inches='tight', dpi=300)
    plt.close()

    return bike_boxes, result_img_path, box_txt_path

# -------------------------- 4. 执行检测并输出结果 --------------------------
if __name__ == "__main__":
    try:
        # 执行检测
        bike_boxes, result_img, box_txt = detect_shared_bike(IMG_PATH, SAVE_DIR)

        # 打印实验结果
        print("="*50)
        print("实验结果输出：")
        print(f"1. 检测到的共享单车数量：{len(bike_boxes)}")
        print(f"2. 共享单车位置坐标：{bike_boxes}")
        print(f"3. 带检测框的图片保存路径：{result_img}")
        print(f"4. 位置数据文本保存路径：{box_txt}")
        print("="*50)

    except Exception as e:

        print(f"检测过程出错：{e}")
