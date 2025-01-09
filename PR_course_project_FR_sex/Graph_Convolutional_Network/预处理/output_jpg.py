import os
import numpy as np
from PIL import Image

def convert_raw_to_jpg(rawdata_dir, jpg_output_dir):
    """
    将原始二进制数据文件转换为 JPG 格式的图像文件。

    参数:
    rawdata_dir (str): 原始数据目录。
    jpg_output_dir (str): 输出 JPG 文件的目录。
    """
    # 创建输出目录（如果不存在）
    os.makedirs(jpg_output_dir, exist_ok=True)

    # 获取所有原始数据文件
    raw_files = [f for f in os.listdir(rawdata_dir) if os.path.isfile(os.path.join(rawdata_dir, f))]

    # 遍历每个文件并将其转换为 JPG 格式
    for raw_file in raw_files:
        try:
            # 构建文件路径
            raw_path = os.path.join(rawdata_dir, raw_file)

            # 读取原始二进制数据
            with open(raw_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)

            # 检查数据长度是否匹配 128x128
            if data.size != 128 * 128:
                print(f"跳过文件 {raw_file}: 数据大小不匹配")
                continue

            # 重塑为 128x128 的灰度图像
            img_array = data.reshape((128, 128))

            # 转换为 PIL 图像对象
            img = Image.fromarray(img_array, mode='L')

            # 保存为 JPG 文件
            output_path = os.path.join(jpg_output_dir, f"{raw_file}.jpg")
            img.save(output_path)
            print(f"已成功转换文件 {raw_file} 为 {output_path}")

        except Exception as e:
            print(f"处理文件 {raw_file} 时出错: {e}")

    print("所有文件处理完成！")

if __name__ == "__main__":
    # 设置原始数据目录和输出目录
    rawdata_dir = "D:/Graph_Convolutional_Network/rawdata/rawdata"
    jpg_output_dir = "output_jpg"

    # 调用函数进行转换
    convert_raw_to_jpg(rawdata_dir, jpg_output_dir)