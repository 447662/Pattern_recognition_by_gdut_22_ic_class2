from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据路径
dataset_path = 'C:/datasets/gender_data'
df = pd.read_csv(dataset_path)

# 分离特征和标签
X = df.drop(columns=['gender']).values
y = df['gender'].values

# 数据预处理：划分训练集和测试集，进行标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 超参数搜索空间
param_grid = {
    'max_depth': [5, 7, 9, 11, 13, None],         # 深度范围
    'min_samples_split': [4, 6, 8, 10],    # 最小分裂样本数
    'min_samples_leaf': [1, 2, 3],         # 叶节点最小样本数
    'criterion': ['gini', 'entropy']       # 划分标准
}

# 使用GridSearchCV进行超参数调优
clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数和模型性能
print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证得分:", grid_search.best_score_)

# 使用最佳参数模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 输出测试集准确率
print("测试集准确率:", accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred, output_dict=True)

# 输出分类报告
print("分类报告:\n", classification_report(y_test, y_pred))

# 4. 可视化分类指标
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
categories = ['Male', 'Female']

# 绘制分类指标曲线图
plt.figure(figsize=(10, 6))
plt.plot(categories, precision, label='Precision', marker='o')
plt.plot(categories, recall, label='Recall', marker='s')
plt.plot(categories, f1, label='F1-Score', marker='^')
plt.title('Classification Metrics by Category')
plt.xlabel('Category')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()

# 绘制支持数条形图
plt.figure(figsize=(10, 6))
plt.bar(categories, support, color=['blue', 'orange'], alpha=0.7)
plt.title('Support by Category')
plt.xlabel('Category')
plt.ylabel('Support Count')
plt.grid(axis='y')
plt.show()
