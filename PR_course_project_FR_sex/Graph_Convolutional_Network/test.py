import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# 超参数配置
config = {
    'input_dim': 3,  # 输入特征维度（3D 坐标）
    'hidden_dim': 512,  # 隐藏层的维度
    'output_dim': 2,  # 输出维度（2：性别男女）
    'batch_size': 32,  # 每个批次的数据量
    'save_dir': 'D:/Graph_Convolutional_Network/saved_models',  # 模型保存路径
}


# 自定义加载数据函数
def load_graph_data_from_directory(directory):
    graph_data_list = []
    for graph_file in os.listdir(directory):
        if graph_file.endswith(".pkl"):
            with open(os.path.join(directory, graph_file), "rb") as f:
                graph_data = pickle.load(f)
                graph_data_list.append(graph_data)
    return graph_data_list


# 定义 GCN 模型
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # 输出一个图的标签

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x


# 测试函数
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 推理时不需要计算梯度
        for data in loader:
            out = model(data)  # 模型输出 logits
            pred = out.argmax(dim=1)  # 获取预测类别
            correct += (pred == data.y).sum().item()  # 与真实标签比较
            total += data.y.size(0)  # 累加样本总数
    return correct / total  # 返回准确率


if __name__ == '__main__':
    # 加载测试数据
    test_data = load_graph_data_from_directory("D://Graph_Convolutional_Network/datas/graphs/test")
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    # 实例化模型并加载训练好的权重
    model = GCNModel(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], output_dim=config['output_dim'])


    # 假设你想加载最好的一个模型
    model_path = "D:/Graph_Convolutional_Network/saved_models/gcn_epoch_600.pth"  # 模型文件路径
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 设置模型为评估模式

    # 测试模型
    accuracy = test(model, test_loader)
    print(f"测试准确率为: {accuracy:.4f}")
