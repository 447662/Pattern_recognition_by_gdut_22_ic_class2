import pickle
import os
import numpy as np
import torch
import mediapipe as mp
from torch_geometric.data import Data
import random

# 初始化 Mediapipe 的人脸模型
mp_face_mesh = mp.solutions.face_mesh
connections = mp_face_mesh.FACEMESH_TESSELATION  # 人脸拓扑结构

def process_npy_files(input_dir, output_dir, max_files=1600):
    """
    处理指定目录下的前 max_files 个 .npy 文件，并将其转换为图数据保存为 .pkl 文件。

    参数:
    input_dir (str): 输入目录，包含 .npy 文件。
    output_dir (str): 输出目录，用于保存 .pkl 文件。
    max_files (int): 要处理的最大 .npy 文件数量。
    """
    os.makedirs(output_dir, exist_ok=True)  # 创建保存图数据的文件夹（如果不存在）

    # 获取所有 .npy 文件
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    # 随机打乱文件列表
    random.shuffle(npy_files)

    # 选择前 max_files 个文件
    selected_files = npy_files[:max_files]

    count = 0
    # 遍历选定的 .npy 文件
    for npy_file in selected_files:
        count += 1
        process_single_npy_file(input_dir, output_dir, npy_file, count)

    print("所有图数据已处理完成！")

def process_single_npy_file(input_dir, output_dir, npy_file, count):
    """
    处理单个 .npy 文件，并将其转换为图数据保存为 .pkl 文件。

    参数:
    input_dir (str): 输入目录，包含 .npy 文件。
    output_dir (str): 输出目录，用于保存 .pkl 文件。
    npy_file (str): 要处理的 .npy 文件名。
    count (int): 当前处理的文件计数。
    """
    # 读取 .npy 文件
    face_4x478 = np.load(os.path.join(input_dir, npy_file))  # 形状: (4, 478)

    # 提取前三行作为 3D 坐标
    face_3x478 = face_4x478[:3, :]

    face_coordinates = face_3x478.T  # 转置为 (478, 3)

    # 节点特征 (Node Features)
    x = torch.tensor(face_coordinates, dtype=torch.float)

    # 将 edges 转为 edge_index
    edges = list(connections)  # 连接关系
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 转置为 (2, E)

    # 标签 (是否男女：0 或 1)，假设标签存储在第 4 行的第 1 列
    label = torch.tensor([int(face_4x478[3, 0])], dtype=torch.long)

    # 构建 PyTorch Geometric 图数据
    graph_data = Data(x=x, edge_index=edge_index, y=label)

    # 保存图数据
    graph_file = os.path.join(output_dir, f"{npy_file.split('.')[0]}.pkl")
    with open(graph_file, "wb") as f:
        pickle.dump(graph_data, f)

    print(f"图数据已保存：{graph_file}", count)

if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "D:/Graph_Convolutional_Network/datas/output_4npy"
    output_dir = "D:/Graph_Convolutional_Network/datas/graphs/output_pkl"

    # 处理前 1600 个 .npy 文件
    process_npy_files(input_dir, output_dir, max_files=1600)