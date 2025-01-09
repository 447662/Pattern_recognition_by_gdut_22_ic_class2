import os
import shutil
import random

def create_directory(path):
    """
    创建目录，如果目录不存在则创建
    :param path: 目录路径
    """
    os.makedirs(path, exist_ok=True)

def get_pkl_files(directory):
    """
    获取目录中的所有 pkl 文件
    :param directory: 目录路径
    :return: pkl 文件列表
    """
    return [f for f in os.listdir(directory) if f.endswith('.pkl')]

def split_files(files, test_size, train_size):
    """
    随机打乱文件列表并分配到测试集和训练集
    :param files: 文件列表
    :param test_size: 测试集大小
    :param train_size: 训练集大小
    :return: 测试集文件列表, 训练集文件列表
    """
    random.shuffle(files)
    return files[:test_size], files[test_size:test_size + train_size]

def copy_files(files, source_dir, target_dir):
    """
    复制文件到目标目录
    :param files: 文件列表
    :param source_dir: 源目录
    :param target_dir: 目标目录
    """
    for file in files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, file))

# 定义文件路径
source_dir = "D:/Graph_Convolutional_Network/datas/graphs/output_pkl"
test_dir = "D:/Graph_Convolutional_Network/datas/graphs/test"
train_dir = "D:/Graph_Convolutional_Network/datas/graphs/train"

# 创建目标文件夹
create_directory(test_dir)
create_directory(train_dir)

# 获取所有的 pkl 文件
pkl_files = get_pkl_files(source_dir)

# 分配文件到测试集和训练集
test_files, train_files = split_files(pkl_files, 150, 1200)

# 复制文件到测试集文件夹
copy_files(test_files, source_dir, test_dir)

# 复制文件到训练集文件夹
copy_files(train_files, source_dir, train_dir)

print(f"已将 {len(test_files)} 个文件复制到测试集文件夹: {test_dir}")
print(f"已将 {len(train_files)} 个文件复制到训练集文件夹: {train_dir}")