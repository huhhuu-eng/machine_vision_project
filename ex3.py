import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ========== 全局环境配置 ==========
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
plt.switch_backend('TkAgg')
zh_font = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
plt.rcParams['axes.unicode_minus'] = False


# ========== 1. 训练/加载模型 ==========
def train_or_load_model():
    model_path = 'my_mnist_model.keras'
    if os.path.exists(model_path):
        print("加载预训练模型...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("首次运行，训练MNIST模型...")
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
        test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=8, batch_size=32, validation_split=0.1)
        model.save(model_path)

    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"模型测试准确率：{test_acc:.4f}")
    return model


# ========== 2. 优化数字分割 ==========
def auto_split_student_id(image_path):
    """
    优化版分割：放宽过滤条件+处理粘连+补充漏检
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    # 预处理：增强对比度+更柔和的形态学操作
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # 自适应二值化
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=8
    )
    # 形态学操作：先膨胀后腐蚀，避免断裂/粘连
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)  # 膨胀：连接断裂笔画
    binary = cv2.erode(binary, kernel, iterations=1)  # 腐蚀：细化粘连部分

    # 检测轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w >= 8 and h >= 15 and 0.3 < (h / w) < 5:
            digit_contours.append((x, y, w, h))

    # 按x坐标排序
    digit_contours = sorted(digit_contours, key=lambda c: c[0])

    # 处理1：粘连数字拆分（宽高比异常的轮廓）
    split_contours = []
    for (x, y, w, h) in digit_contours:
        if w / h > 1.5:  # 宽度远大于高度，判定为粘连
            # 拆分为两个数字
            split_w = w // 2
            split_contours.append((x, y, split_w, h))
            split_contours.append((x + split_w, y, split_w, h))
        else:
            split_contours.append((x, y, w, h))

    # 重新排序
    split_contours = sorted(split_contours, key=lambda c: c[0])

    # 处理2：补充漏检（不足10个时，按间距填充）
    final_contours = split_contours[:10]
    if len(final_contours) < 10:
        print(f"⚠️  检测到{len(final_contours)}个数字")
        # 计算平均间距，补充漏检位置
        if len(final_contours) >= 2:
            total_width = final_contours[-1][0] + final_contours[-1][2] - final_contours[0][0]
            avg_gap = total_width / 10  # 平均每个数字的宽度
            # 按平均间距生成漏检位置
            start_x = final_contours[0][0]
            for i in range(10):
                if i >= len(final_contours):
                    # 补充漏检的轮廓位置
                    x = int(start_x + i * avg_gap)
                    y = final_contours[0][1]
                    w = int(avg_gap * 0.8)
                    h = final_contours[0][3]
                    final_contours.append((x, y, w, h))

        final_contours = final_contours[:10]

    # 分割每个数字
    digit_imgs = []
    id_region = img.copy()
    for (x, y, w, h) in final_contours:
        expand = int(max(w, h) * 0.1)
        x1 = max(0, x - expand)
        y1 = max(0, y - expand)
        x2 = min(img.shape[1], x + w + expand)
        y2 = min(img.shape[0], y + h + expand)
        digit = img[y1:y2, x1:x2]
        digit_imgs.append(digit)
        cv2.rectangle(id_region, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return digit_imgs, id_region


# ========== 3. 优化预处理+校准 ==========
def optimize_preprocess(digit_img):
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    # 细化笔画
    kernel_erode = np.ones((2, 2), np.uint8)
    binary = cv2.erode(binary, kernel_erode, iterations=1)
    # 填充0的孔洞
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            cv2.drawContours(binary, [cnt], 0, 255, -1)
    resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_CUBIC)
    return resized.astype("float32") / 255


def calibrate_prediction(pred_label, processed_digit):
    # 校准4→7（7有右上斜杠，4无）
    if pred_label == 4:
        slash_area = np.sum(processed_digit[10:20, 20:28])
        if slash_area > 5:
            return 7
    # 校准2→0（0是闭合轮廓，2是开放）
    if pred_label == 2:
        hole_area = np.sum(processed_digit[8:20, 8:20])
        if hole_area < 3:
            return 0
    # 校准8→0（0无中间分隔线）
    if pred_label == 8:
        line_area = np.sum(processed_digit[14:16, 8:20])
        if line_area < 1:
            return 0
    return pred_label


# ========== 4. 预测+校准 ==========
def predict_digits(digit_imgs, model):
    processed_digits = []
    results = []
    confidences = []
    for img in digit_imgs:
        processed = optimize_preprocess(img)
        processed_digits.append(processed)
        pred = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)
        pred_label = np.argmax(pred)
        pred_conf = np.max(pred)
        calibrated_label = calibrate_prediction(pred_label, processed)
        results.append(calibrated_label)
        confidences.append(pred_conf)
    return results, confidences, processed_digits


# ========== 5. 可视化 ==========
def visualize_results(digit_imgs, processed_digits, results, confidences):
    fig, axes = plt.subplots(2, 10, figsize=(20, 8))
    # 第一行：原始数字
    for i in range(10):
        axes[0, i].imshow(cv2.cvtColor(digit_imgs[i], cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"原始第{i + 1}位", fontproperties=zh_font)
        axes[0, i].axis("off")
    # 第二行：预测结果
    for i in range(10):
        axes[1, i].imshow(processed_digits[i], cmap="gray")
        color = "green" if confidences[i] > 0.8 else "orange"
        axes[1, i].set_title(
            f"预测：{results[i]} ({confidences[i]:.2f})",
            fontproperties=zh_font, color=color
        )
        axes[1, i].axis("off")
    # 最终结果
    student_id = "".join(map(str, results))
    plt.suptitle(
        f"最终识别学号：{student_id}（平均置信度：{np.mean(confidences):.3f}）",
        fontproperties=zh_font, fontsize=16
    )
    plt.tight_layout()
    plt.savefig("final_id_recognition.png", dpi=150, bbox_inches="tight")
    plt.show()
    return student_id


# ========== 主程序 ==========
if __name__ == "__main__":
    ID_PHOTO_PATH = "id_photo.jpg"

    try:
        print("===== 手写学号识别 =====")
        # 1. 加载模型
        model = train_or_load_model()

        # 2
        print("\n 自动分割学号数字")
        digit_imgs, id_region = auto_split_student_id(ID_PHOTO_PATH)
        print(f"成功分割出{len(digit_imgs)}个数字")

        # 3. 预测+校准
        print("\n识别并校准数字...")
        results, confidences, processed_digits = predict_digits(digit_imgs, model)

        # 4. 可视化
        student_id = visualize_results(digit_imgs, processed_digits, results, confidences)

        # 输出结果
        print(f"\n 识别完成！")
        print(f"   识别学号：{student_id}")
        print(f"   置信度列表：{[f'{c:.3f}' for c in confidences]}")
        print(f"   生成文件：final_id_recognition.png、id_region_with_contours.png")
        cv2.imwrite("id_region_with_contours.png", id_region)

    except FileNotFoundError as e:
        print(f"\n错误：{e}")
        print("   解决：检查图片路径/格式")