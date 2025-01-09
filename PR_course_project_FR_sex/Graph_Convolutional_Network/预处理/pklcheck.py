import os
import pickle

# 设置要检查的目录
pkl_dir = "D:/Graph_Convolutional_Network/datas/graphs/output_pkl"

# 获取目录中的所有 .pkl 文件，排除 output 文件夹
pkl_files = []
for root, dirs, files in os.walk(pkl_dir):
    # 排除 output 文件夹
    if 'output' in dirs:
        dirs.remove('output')
    # 只检查 test 和 train 文件夹
    if 'test' in root or 'train' in root:
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))

# 遍历每个 .pkl 文件并检查其内容
for pkl_path in pkl_files:
    try:
        # 加载 .pkl 文件
        with open(pkl_path, "rb") as f:
            loaded_graph_data = pickle.load(f)

        # 检查加载后的图数据
        print(f"文件名称: {os.path.basename(pkl_path)}")
        print("节点特征 (x):", loaded_graph_data.x)
        print("边索引 (edge_index):", loaded_graph_data.edge_index)
        print("标签 (y):", loaded_graph_data.y)
        print("-" * 50)
    except Exception as e:
        print(f"无法加载文件 {os.path.basename(pkl_path)}: {e}")