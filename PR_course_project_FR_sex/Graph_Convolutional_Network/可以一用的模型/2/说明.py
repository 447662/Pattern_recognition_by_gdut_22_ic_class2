# 本模型超参数为
# 超参数配置
config = {
    'input_dim': 3,  # 输入特征维度（3D 坐标）
    'hidden_dim': 256,  # 隐藏层的维度
    'output_dim': 2,  # 输出维度（2：性别男女）
    'learning_rate': 0.002,  # 初始学习率
    'batch_size': 32,  # 每个批次的数据量
    'epochs': 600,  # 训练的轮数
    'save_interval': 5,  # 每几个epoch保存一次模型
    'print_every_sample': 5,  # 每几个样本汇报一次效果
    'save_dir': 'D:/桌面/homework/patternrecognition/Graph_Convolutional_Network/saved_models',  # 模型保存路径
    'log_file': 'D:/桌面/homework/patternrecognition/Graph_Convolutional_Network/training.txt',  # 训练日志文件路径
    'threshold': 0.5,  # 判断为正样本的置信度
    'lr_stable_epochs': 400,  # 固定学习率的轮数
    'lr_decay_epochs': 200,  # 学习率衰减的轮数
    'early_stopping_patience': 80,  # 早停法的耐心值
    'early_stopping_start_epoch': 200,  # 从第200轮开始使用早停法
}
