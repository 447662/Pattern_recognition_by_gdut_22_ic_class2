import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 黑体字体

# 读取训练日志文件并解析其中的 Loss 和 Accuracy
def parse_training_log(log_file_path):
    """
    解析训练日志文件，提取每个 epoch 的损失和准确率
    :param log_file_path: 日志文件路径
    :return: epochs, losses, accuracies 列表
    """
    epochs = []
    losses = []
    accuracies = []

    with open(log_file_path, 'r') as file:
        for line in file:
            # 每一行都应该包含类似于 "Epoch X, Loss: Y, Accuracy: Z, lr: W"
            if 'Epoch' in line:
                parts = line.strip().split(', ')

                if len(parts) >= 4:
                    epoch = int(parts[0].split(' ')[1])  # 提取 Epoch 数字
                    loss = float(parts[1].split(' ')[1])  # 提取 Loss 数字
                    accuracy = float(parts[2].split(' ')[1])  # 提取 Accuracy 数字
                    print(epoch, loss, accuracy)

                    epochs.append(epoch)
                    losses.append(loss)
                    accuracies.append(accuracy)

    return epochs, losses, accuracies

if __name__ == '__main__':
    # 从日志文件中读取数据
    log_file_path = 'D:/Graph_Convolutional_Network/training_log.txt'

    epochs, losses, accuracies = parse_training_log(log_file_path)

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 6))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Loss", color='r', marker='o')
    plt.title('训练集损失', fontproperties=font)
    plt.xlabel('轮数', fontproperties=font)
    plt.ylabel('损失', fontproperties=font)
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label="Accuracy", color='g', marker='o')
    plt.title('模型准确率', fontproperties=font)
    plt.xlabel('轮数', fontproperties=font)
    plt.ylabel('准确率', fontproperties=font)
    plt.grid(True)

    # 显示图形
    plt.tight_layout()
    plt.show()