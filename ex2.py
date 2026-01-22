import cv2
import numpy as np


def process_image(image_path, output_path=None):

    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    original_image = image.copy()

    # 2. 图像预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊，减少噪声
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测 (Canny算法)
    edges = cv2.Canny(blur, 50, 150)

    # 3. 兴趣区域提取
    mask = np.zeros_like(edges)

    # 定义多边形区域（假设车道线在图像下方的梯形区域）
    height, width = edges.shape
    vertices = np.array([
        [(0, height), (width / 2 - 50, height / 2 + 60), (width / 2 + 50, height / 2 + 60), (width, height)]
    ], dtype=np.int32)

    # 填充多边形区域
    cv2.fillPoly(mask, vertices, 255)

    # 应用掩码，只保留兴趣区域内的边缘
    masked_edges = cv2.bitwise_and(edges, mask)

    # 4. 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,  # 极坐标的rho分辨率，1像素
        theta=np.pi / 180,  # 极坐标的theta分辨率，1度
        threshold=50,  # 检测阈值
        minLineLength=100,  # 最小直线长度
        maxLineGap=50  # 最大直线间隙
    )

    # 5. 直线分类与平均
    left_lines = []  # 左侧车道线
    right_lines = []  # 右侧车道线

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算直线斜率
            if x2 - x1 == 0:
                slope = float('inf')
            else:
                slope = (y2 - y1) / (x2 - x1)

            # 根据斜率分类直线
            # 左侧车道线通常斜率为负，右侧车道线通常斜率为正
            if abs(slope) > 0.5:  # 过滤掉几乎水平的直线
                if slope < 0:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])

    # 平均左侧车道线
    left_line = average_lines(left_lines, height)
    # 平均右侧车道线
    right_line = average_lines(right_lines, height)

    # 6. 绘制车道线
    result = draw_lanes(original_image, [left_line, right_line])

    # 7. 保存结果图像
    if output_path is not None:
        cv2.imwrite(output_path, result)
        print(f"结果已保存到: {output_path}")

    return result


def average_lines(lines, height):

    if not lines:
        return None

    # 计算所有直线的斜率和截距
    slopes = []
    intercepts = []

    for x1, y1, x2, y2 in lines:
        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        slopes.append(slope)
        intercepts.append(intercept)

    # 计算平均斜率和截距
    avg_slope = np.mean(slopes)
    avg_intercept = np.mean(intercepts)

    # 计算直线在图像中的起点和终点
    # 假设车道线从图像底部开始，到图像高度的60%处结束
    y1 = height
    y2 = int(height * 0.6)

    # 计算对应的x坐标
    x1 = int((y1 - avg_intercept) / avg_slope)
    x2 = int((y2 - avg_intercept) / avg_slope)

    return [x1, y1, x2, y2]


def draw_lanes(image, lines):

    result = image.copy()

    # 创建一个空白图像，用于绘制车道线
    lane_image = np.zeros_like(result)

    # 绘制每条车道线
    for line in lines:
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    # 将车道线图像与原始图像合并
    result = cv2.addWeighted(result, 0.8, lane_image, 1, 0)

    return result


def main():

    input_path = "campus_road.jpg"
    output_path = "2.result.jpg"

    # 处理图像
    result = process_image(input_path, output_path)
    if result is not None:
        # 显示结果
        cv2.imshow('Lane Detection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()