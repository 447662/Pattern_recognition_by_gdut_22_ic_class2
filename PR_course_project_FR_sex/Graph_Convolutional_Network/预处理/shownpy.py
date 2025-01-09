import numpy as np

# 加载 .npy 文件
file_npy = "D:/Graph_Convolutional_Network/datasets/output_3npy/1223.npy"
data = np.load(file_npy)

# 打印数组内容和形状
print("data shape:", data.shape)
print("data:", data)
