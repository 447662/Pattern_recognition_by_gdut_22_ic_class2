import matplotlib.pyplot as plt
from torch_geometric.data import Data
import numpy as np
import mediapipe as mp
import torch

def load_face_data(npy_file):
    """
    加载人脸关键点的 3D 坐标和标签。

    参数:
    npy_file (str): .npy 文件路径。

    返回:
    tuple: 包含 3D 坐标和标签的元组。
    """
    face_4x478 = np.load(npy_file)  # 形状: (4, 478)
    face_3x478 = face_4x478[:3, :]  # 提取前三行作为 3D 坐标
    face_coordinates = face_3x478.T  # 转置为 (478, 3)
    label = int(face_4x478[3, 0])  # 提取标签
    return face_coordinates, label

def create_graph_data(face_coordinates, label):
    """
    创建 PyTorch Geometric 图数据。

    参数:
    face_coordinates (ndarray): 人脸关键点的 3D 坐标。
    label (int): 标签 (是否微笑：0 或 1)。

    返回:
    Data: PyTorch Geometric 图数据。
    """
    x = torch.tensor(face_coordinates, dtype=torch.float)  # 节点特征 (Node Features)
    
    # 初始化 Mediapipe 的人脸模型
    mp_face_mesh = mp.solutions.face_mesh
    connections = mp_face_mesh.FACEMESH_TESSELATION  # 人脸拓扑结构
    
    # 提取连接关系 (edges)
    edges = list(connections)  # connections 是一个集合，每个元素是 (start, end)
    
    # 将 edges 转为 edge_index
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 转置为 (2, E)
    
    # 构建 PyTorch Geometric 图数据
    graph_data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    return graph_data

def visualize_3d_graph(graph_data):
    """
    可视化 3D 图数据。

    参数:
    graph_data (Data): PyTorch Geometric 图数据。
    """
    # 提取节点坐标 (x, y, z)
    node_coords = graph_data.x.numpy()  # 转为 NumPy 数组
    edge_index = graph_data.edge_index.numpy()  # 转为 NumPy 数组

    # 创建一个 3D 图形
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制节点
    ax.scatter(
        node_coords[:, 0],  # x 坐标
        node_coords[:, 1],  # y 坐标
        node_coords[:, 2],  # z 坐标
        c='green',           # 节点颜色
        s=20,               # 节点大小
        label='Nodes'
    )

    # 绘制边
    for start, end in edge_index.T:  # 遍历每条边
        x_coords = [node_coords[start, 0], node_coords[end, 0]]
        y_coords = [node_coords[start, 1], node_coords[end, 1]]
        z_coords = [node_coords[start, 2], node_coords[end, 2]]

        ax.plot(
            x_coords, y_coords, z_coords, c='gray', alpha=0.5, linewidth=0.5
        )

    # 添加标题和图例
    ax.set_title("3D Face Mesh", fontsize=16)
    ax.legend()
    
    # 调整视角
    ax.view_init(elev=-85, azim=-90)
    
    plt.show()

if __name__ == "__main__":
    # 假设你已经提取了人脸关键点的 3D 坐标和边
    npy_file = "D:/Graph_Convolutional_Network/datas/output_4npy/1224.npy"
    face_coordinates, label = load_face_data(npy_file)
    graph_data = create_graph_data(face_coordinates, label)
    visualize_3d_graph(graph_data)