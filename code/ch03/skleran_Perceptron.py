import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.metrics import accuracy_score
from plot import plot_decision_regions

# 加载数据集
iris = datasets.load_iris()
# 取第三四列数据作为样本
X = iris.data[:, [2, 3]]
# 取类标向量
y = iris.target
# 随机将数据矩阵X与类标向量y按照7:3的比例划分为训练数据集和测试数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
# 对特征进行标准化处理
sc = StandardScaler()
sc.fit(X_train)  # 计算训练样本中每个特征的μ（样本均值）和σ（标准差）
X_train_std = sc.transform(X_train)  # 使用前面计算的μ（样本均值）和σ（标准差）来对训练数据做标准化处理
X_test_std = sc.transform(X_test)
# 训练数据
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
# 预测数据
y_pred = ppn.predict(X_test_std)
#分类准确率：正确分类的个数占所有测试数据总数的比率
print('准确率：%.2f' % accuracy_score(y_test,y_pred))

plot_decision_regions(X_train_std,y_train,classifier=ppn)
plt.xlabel('perceptron')
plt.show()

# 使用逻辑回归模型
lr=LogisticRegression(C=1000.0,random_state=0) #C是正则化系数的倒数！
lr.fit(X_train_std,y_train)
plot_decision_regions(X_train_std,y_train,classifier=lr)
plt.xlabel('logistic regression')
plt.show()
