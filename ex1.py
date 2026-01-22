import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.backends.backend_agg as backend_agg

matplotlib.use('Agg')

# 配置Matplotlib中文支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.dpi'] = 100


class ImageFilteringExperiment:
    def __init__(self, image_path):
        """初始化实验类"""
        self.image_path = image_path
        self.image = None
        self.gray_image = None
        self.sobel_x = None
        self.sobel_y = None
        self.sobel_combined = None
        self.custom_filtered = None
        self.color_histograms = None
        self.texture_features = None

    def read_image(self):
        """读取图像并转换为numpy数组"""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {self.image_path}")

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # 转换为灰度图
        self.gray_image = self.rgb_to_gray(self.image)

    def rgb_to_gray(self, rgb_image):
        """将RGB图像转换为灰度图"""
        gray = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
        return gray.astype(np.uint8)

    def convolve(self, image, kernel):
        """自定义卷积函数"""
        # 获取图像和核的尺寸
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # 计算填充大小
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # 创建填充后的图像
        padded_image = np.zeros((image_height + 2 * pad_height, image_width + 2 * pad_width), dtype=np.float32)
        padded_image[pad_height:pad_height + image_height, pad_width:pad_width + image_width] = image.astype(np.float32)

        # 初始化结果图像
        result = np.zeros_like(image, dtype=np.float32)

        # 执行卷积
        for i in range(image_height):
            for j in range(image_width):
                # 提取感兴趣区域
                roi = padded_image[i:i + kernel_height, j:j + kernel_width]
                # 执行卷积运算
                result[i, j] = np.sum(roi * kernel)

        return result

    def sobel_filter(self):
        """应用Sobel算子"""
        # Sobel X 方向卷积核
        sobel_x_kernel = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=np.float32)

        # Sobel Y 方向卷积核
        sobel_y_kernel = np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=np.float32)

        # 应用卷积
        self.sobel_x = self.convolve(self.gray_image, sobel_x_kernel)
        self.sobel_y = self.convolve(self.gray_image, sobel_y_kernel)

        # 计算合并的Sobel结果
        self.sobel_combined = np.sqrt(self.sobel_x ** 2 + self.sobel_y ** 2)

        # 归一化到0-255
        self.sobel_x = self.normalize(self.sobel_x)
        self.sobel_y = self.normalize(self.sobel_y)
        self.sobel_combined = self.normalize(self.sobel_combined)

    def custom_filter(self):
        """应用给定的卷积核"""
        # 给定的卷积核
        custom_kernel = np.array([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]], dtype=np.float32)

        # 应用卷积
        self.custom_filtered = self.convolve(self.gray_image, custom_kernel)

        # 归一化到0-255
        self.custom_filtered = self.normalize(self.custom_filtered)

    def normalize(self, image):
        """将图像归一化到0-255"""
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val == 0:
            return np.zeros_like(image, dtype=np.uint8)
        normalized = 255 * ((image - min_val) / (max_val - min_val))
        return normalized.astype(np.uint8)

    def calculate_color_histograms(self):
        """计算颜色直方图"""
        self.color_histograms = {}
        channels = ['R', 'G', 'B']

        for i, channel in enumerate(channels):
            # 计算每个颜色通道的直方图
            hist = np.zeros(256, dtype=np.int32)
            for pixel in self.image[..., i].flatten():
                hist[pixel] += 1
            self.color_histograms[channel] = hist

    def extract_texture_features(self):
        """提取纹理特征（使用灰度共生矩阵）"""
        # 将灰度图量化为16个等级，减少计算量
        quantized = (self.gray_image // 16).astype(np.uint8)

        # 计算灰度共生矩阵（0度方向，距离1）
        glcm = self.calculate_glcm(quantized, distance=1, angle=0)

        # 提取纹理特征
        self.texture_features = {
            'contrast': self.calculate_contrast(glcm),
            'correlation': self.calculate_correlation(glcm),
            'energy': self.calculate_energy(glcm),
            'homogeneity': self.calculate_homogeneity(glcm)
        }

    def calculate_glcm(self, image, distance=1, angle=0):
        """计算灰度共生矩阵"""
        levels = 16  # 量化后的灰度等级
        glcm = np.zeros((levels, levels), dtype=np.int32)

        height, width = image.shape

        # 根据角度计算偏移
        if angle == 0:
            dy, dx = 0, distance
        elif angle == 45:
            dy, dx = -distance, distance
        elif angle == 90:
            dy, dx = -distance, 0
        elif angle == 135:
            dy, dx = -distance, -distance
        else:
            dy, dx = 0, distance

        # 计算共生矩阵
        for i in range(height):
            for j in range(width):
                row = image[i, j]
                # 检查相邻像素是否在图像范围内
                if 0 <= i + dy < height and 0 <= j + dx < width:
                    col = image[i + dy, j + dx]
                    glcm[row, col] += 1

        return glcm

    def calculate_contrast(self, glcm):
        """计算对比度"""
        levels = glcm.shape[0]
        contrast = 0
        for i in range(levels):
            for j in range(levels):
                contrast += (i - j) ** 2 * glcm[i, j]
        return contrast

    def calculate_correlation(self, glcm):
        """计算相关性"""
        levels = glcm.shape[0]

        # 计算行列均值
        row_mean = np.zeros(levels)
        col_mean = np.zeros(levels)
        total = np.sum(glcm)

        if total == 0:
            return 0

        for i in range(levels):
            row_sum = np.sum(glcm[i, :])
            row_mean[i] = i * row_sum / total

            col_sum = np.sum(glcm[:, i])
            col_mean[i] = i * col_sum / total

        overall_mean = np.sum(row_mean)

        # 计算标准差
        row_std = 0
        for i in range(levels):
            row_sum = np.sum(glcm[i, :])
            row_std += (i - overall_mean) ** 2 * row_sum / total
        row_std = np.sqrt(row_std)

        col_std = row_std  # 假设行列分布相同

        if row_std == 0 or col_std == 0:
            return 0

        # 计算相关性
        correlation = 0
        for i in range(levels):
            for j in range(levels):
                correlation += ((i - overall_mean) * (j - overall_mean) * glcm[i, j]) / total

        correlation /= (row_std * col_std)
        return correlation

    def calculate_energy(self, glcm):
        """计算能量"""
        return np.sum(glcm ** 2)

    def calculate_homogeneity(self, glcm):
        """计算同质性"""
        levels = glcm.shape[0]
        homogeneity = 0
        for i in range(levels):
            for j in range(levels):
                homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
        return homogeneity

    def visualize_results(self):
        """可视化所有结果"""
        # 创建Figure对象
        fig = plt.figure(figsize=(15, 12))

        # 原始图像
        plt.subplot(3, 3, 1)
        plt.imshow(self.image)
        plt.title('原始图像')
        plt.axis('off')

        # 灰度图像
        plt.subplot(3, 3, 2)
        plt.imshow(self.gray_image, cmap='gray')
        plt.title('灰度图像')
        plt.axis('off')

        # Sobel X
        plt.subplot(3, 3, 4)
        plt.imshow(self.sobel_x, cmap='gray')
        plt.title('Sobel X方向')
        plt.axis('off')

        # Sobel Y
        plt.subplot(3, 3, 5)
        plt.imshow(self.sobel_y, cmap='gray')
        plt.title('Sobel Y方向')
        plt.axis('off')

        # Sobel 合并
        plt.subplot(3, 3, 6)
        plt.imshow(self.sobel_combined, cmap='gray')
        plt.title('Sobel 合并结果')
        plt.axis('off')

        # 给定卷积核结果
        plt.subplot(3, 3, 7)
        plt.imshow(self.custom_filtered, cmap='gray')
        plt.title('给定卷积核滤波结果')
        plt.axis('off')

        # 颜色直方图
        plt.subplot(3, 3, 3)
        for channel, hist in self.color_histograms.items():
            plt.plot(hist, label=channel)
        plt.title('颜色直方图')
        plt.xlabel('灰度值')
        plt.ylabel('像素数量')
        plt.legend()

        plt.tight_layout()

        # 保存图像
        plt.savefig('result_visualization.png', dpi=300, bbox_inches='tight')

        # 尝试显示图像，使用try-except捕获可能的错误
        try:
            plt.show()
        except AttributeError as e:
            print(f"显示图像时出现错误: {e}")
            print("图像已保存为 result_visualization.png")
        finally:
            plt.close(fig)

    def save_results(self):
        """保存处理后的图像结果"""
        # 转换为BGR格式保存
        cv2.imwrite('sobel_x.jpg', self.sobel_x)
        cv2.imwrite('sobel_y.jpg', self.sobel_y)
        cv2.imwrite('sobel_combined.jpg', self.sobel_combined)
        cv2.imwrite('custom_filtered.jpg', self.custom_filtered)

        # 保存原始图像（BGR格式）
        cv2.imwrite('original.jpg', cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

        # 保存灰度图像
        cv2.imwrite('gray_image.jpg', self.gray_image)

    def print_texture_features(self):
        """打印纹理特征"""
        print("纹理特征：")
        for feature, value in self.texture_features.items():
            print(f"{feature}: {value}")

    def run_experiment(self):
        """运行完整实验"""
        print("开始图像滤波实验...")

        # 读取图像
        self.read_image()
        print("图像读取完成")

        # Sobel滤波
        self.sobel_filter()
        print("Sobel算子滤波完成")

        # 给定卷积核滤波
        self.custom_filter()
        print("给定卷积核滤波完成")

        # 计算颜色直方图
        self.calculate_color_histograms()
        print("颜色直方图计算完成")

        # 提取纹理特征
        self.extract_texture_features()
        print("纹理特征提取完成")

        # 可视化结果
        self.visualize_results()
        print("结果可视化完成")

        # 保存结果
        self.save_results()
        print("结果保存完成")

        # 打印纹理特征
        self.print_texture_features()

        print("实验完成！")


# 主函数
if __name__ == "__main__":

    image_path = "picture1.jpg"

    try:
        experiment = ImageFilteringExperiment(image_path)
        experiment.run_experiment()
    except Exception as e:
        print(f"实验过程中出现错误: {e}")