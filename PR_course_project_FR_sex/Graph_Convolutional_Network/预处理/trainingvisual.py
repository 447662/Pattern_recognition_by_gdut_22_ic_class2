import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 这里使用的是黑体字体，你可以根据需要更改字体路径

# 读取训练日志文件并解析其中的 Loss 和 Accuracy
def parse_training_log(log_file_path):
    """
    解析训练日志文件，提取每个 epoch 的损失和准确率
    :param log_file_path: 日志文件路径
    :return: epochs, train_losses, train_accuracies, test_accuracies 列表
    """
    epochs = []
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    with open(log_file_path, 'r') as file:
        for line in file:
            # 每一行都应该包含类似于 "Epoch X, Loss: Y, Train Accuracy: Z, Test Accuracy: W, lr: V"
            if 'Epoch' in line:
                parts = line.strip().split(', ')

                if len(parts) >= 5:
                    try:
                        epoch = int(parts[0].split(' ')[1])  # 提取 Epoch 数字
                        train_loss = float(parts[1].split(' ')[1])  # 提取 Loss 数字
                        train_accuracy = float(parts[2].split(' ')[2])  # 提取 Train Accuracy 数字
                        test_accuracy = float(parts[3].split(' ')[2])  # 提取 Test Accuracy 数字
                        print(epoch, train_loss, train_accuracy, test_accuracy)

                        epochs.append(epoch)
                        train_losses.append(train_loss)
                        train_accuracies.append(train_accuracy)
                        test_accuracies.append(test_accuracy)
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing line: {line.strip()}")
                        print(e)

    return epochs, train_losses, train_accuracies, test_accuracies

if __name__ == '__main__':
    # 从日志文件中读取数据
    log_file_path = 'D:/桌面/homework/patternrecognition/Graph_Convolutional_Network/training.txt'

    epochs, train_losses, train_accuracies, test_accuracies = parse_training_log(log_file_path)

    # 绘制损失和准确率曲线
    plt.figure(figsize=(18, 6))

    # 绘制 Train Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color='r', marker='o', linestyle='-')
    plt.title('训练集损失', fontproperties=font)
    plt.xlabel('轮数', fontproperties=font)
    plt.ylabel('损失', fontproperties=font)
    plt.grid(True)
    plt.legend()

    # 绘制 Train Accuracy 曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy", color='g', marker='o', linestyle='-')
    plt.title('训练集准确率', fontproperties=font)
    plt.xlabel('轮数', fontproperties=font)
    plt.ylabel('准确率', fontproperties=font)
    plt.grid(True)
    plt.legend()

    # 绘制 Test Accuracy 曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, label="Test Accuracy", color='b', marker='o', linestyle='-')
    plt.title('测试集准确率', fontproperties=font)
    plt.xlabel('轮数', fontproperties=font)
    plt.ylabel('准确率', fontproperties=font)
    plt.grid(True)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()