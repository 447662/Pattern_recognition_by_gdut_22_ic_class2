import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def load_and_aggregate_features(file_path):
    with open(file_path, 'rb') as f:
        graph_data = pickle.load(f)

    # 假设你想使用节点特征作为输入特征
    features = graph_data.x.numpy()  # 转换为NumPy数组
    label = graph_data.y.item()  # 获取标量标签

    # 使用简单的聚合操作，例如取平均值
    aggregated_features = np.mean(features, axis=0)  # 或者使用 np.max 或 np.min

    return aggregated_features, label


# 设置数据路径
data_dir = r"D:\FACE\graphs"

# 加载并聚合所有文件的特征和标签
all_features = []
all_labels = []

for file_name in os.listdir(data_dir):
    if file_name.endswith('.pkl'):
        file_path = os.path.join(data_dir, file_name)
        try:
            features, label = load_and_aggregate_features(file_path)
            all_features.append(features)
            all_labels.append(label)
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

# 将列表转换为NumPy数组
X = np.array(all_features)
y = np.array(all_labels)

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 定义类别标签映射
label_names = {0: '女', 1: '男'}

# 将数字标签映射为文字标签
y_test_named = np.vectorize(label_names.get)(y_test)

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),
    'class_weight': [None, 'balanced']
}

# 创建SVM分类器
clf = svm.SVC(probability=True)

# 使用GridSearchCV进行超参数调优
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数训练模型
best_clf = grid_search.best_estimator_
best_clf.fit(X_train_resampled, y_train_resampled)

# 预测并评估
y_pred_best = best_clf.predict(X_test)
y_pred_best_named = np.vectorize(label_names.get)(y_pred_best)

print("分类报告：")
print(classification_report(y_test_named, y_pred_best_named, labels=['女', '男'], zero_division=0))

# 可选：输出混淆矩阵
conf_matrix = confusion_matrix(y_test_named, y_pred_best_named, labels=['女', '男'])
print("\n混淆矩阵：")
print(conf_matrix)

# 可选：绘制ROC曲线并调整决策阈值
y_prob = best_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 根据需要调整决策阈值
best_threshold = 0.5  # 你可以根据ROC曲线选择一个合适的阈值
y_pred_adjusted = (y_prob >= best_threshold).astype(int)
y_pred_adjusted_named = np.vectorize(label_names.get)(y_pred_adjusted)

print("调整决策阈值后的分类报告：")
print(classification_report(y_test_named, y_pred_adjusted_named, labels=['女', '男'], zero_division=0))