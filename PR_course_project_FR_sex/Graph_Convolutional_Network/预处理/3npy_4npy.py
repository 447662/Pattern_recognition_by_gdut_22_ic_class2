import csv
import numpy as np
import os
import chardet

def detect_file_encoding(file_path, num_bytes=100):
    """
    检测文件的编码格式。

    参数:
    file_path (str): 文件路径。
    num_bytes (int): 用于检测编码的字节数。

    返回:
    str: 文件的编码格式。
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
        result = chardet.detect(raw_data)
        return result['encoding']

def process_csv_file(csv_file, output_dir, npy_dir):
    """
    处理 CSV 文件，将标签添加到对应的 .npy 文件中，并保存到输出目录。

    参数:
    csv_file (str): CSV 文件路径。
    output_dir (str): 输出目录，用于保存更新后的 .npy 文件。
    npy_dir (str): 存储原始 .npy 文件的目录。
    """
    os.makedirs(output_dir, exist_ok=True)  # 创建保存图数据的文件夹（如果不存在）

    encoding = detect_file_encoding(csv_file)

    with open(csv_file, mode='r', encoding=encoding) as file:
        reader = csv.reader(file)

        # 计算总行数并重置文件指针
        total_rows = sum(1 for _ in reader)
        file.seek(0)  # 重置文件指针
        print(f"CSV 总行数: {total_rows}")

        row_count = 0
        for row in reader:
            # 跳过无效行
            if len(row) < 2:
                print(f"跳过无效行: {row}")
                continue

            row_count += 1
            index, label = row[0], row[1]

            npy_file_path = os.path.join(npy_dir, f"{index}.npy")
            npy_file_path_updated = os.path.join(output_dir, f"{index}.npy")

            if not os.path.exists(npy_file_path):
                print(f"文件 {npy_file_path} 不存在！")
            else:
                # 加载 .npy 文件
                data = np.load(npy_file_path)  # 原始数据为 (3, 478)

                # 创建要添加的行
                try:
                    label = float(label)  # 转换标签为数值
                    additional_row = np.full((1, data.shape[1]), label)  # 形状为 (1, 478)

                    # 拼接到第 4 行
                    updated_data = np.vstack((data, additional_row))  # 拼接后为 (4, 478)

                    # 保存更新后的 .npy 文件
                    np.save(npy_file_path_updated, updated_data)
                except ValueError:
                    print(f"标签转换失败: {label}, 跳过此行")

            print(f"{row_count}/{total_rows}")

if __name__ == "__main__":
    # 定义输入输出路径
    csv_file = "D:/Graph_Convolutional_Network/datas/facecsv/face.csv"
    output_dir = "D:/Graph_Convolutional_Network/datas/output_4npy"
    npy_dir = "D:/Graph_Convolutional_Network/datas/output_3npy"

    # 处理 CSV 文件
    process_csv_file(csv_file, output_dir, npy_dir)