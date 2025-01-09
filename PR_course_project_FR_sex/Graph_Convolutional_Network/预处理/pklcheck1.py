import pickle

# 假设你要检查保存的图数据
graph_file = "D:/Graph_Convolutional_Network/datasets/graphs/else/test/1552.pkl" 

# 加载 .pkl 文件
with open(graph_file, "rb") as f:
    loaded_graph_data = pickle.load(f)

# 检查加载后的图数据
print("节点特征 (x):", loaded_graph_data.x)
print("边索引 (edge_index):", loaded_graph_data.edge_index)
print("标签 (y):", loaded_graph_data.y)
