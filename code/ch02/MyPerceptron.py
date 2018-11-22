#!/usr/bin/env python
# coding:utf-8
# @Time    : 2018/11/21 17:48
# @Author  : fanhaiyang
# @File    : MyPerceptron.py
# @Software: PyCharm
# @Desc    : 实现一个感知器
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyPerceptron:
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta  # 学习速率
        self.n_iter = n_iter  # 迭代次数
        self.w_ = []  # 特征权重
        self.errors_ = []  # 分类错误的样本

    def fit(self, X, y):
        # X 所有的样本特征向量，y 类标
        self.w_ = np.zeros(1 + X.shape[1])  # 初始化权重集合为0，长度是特征数加1，因为还有一个初始项
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):  # xi是一个样本的所有特征，target是样本标签
                update = self.eta * (target - self.predict(xi))
                self.w_[0] += update
                self.w_[1:] += update * xi
                errors += int(update != 0.0)  # 该样本如果分类错误，errors+1
            self.errors_.append(errors)
        return self

    def predict(self, X):  # 这里的X就是一个样本的特征的向量
        return np.where((np.dot(X, self.w_[1:]) + self.w_[0]) >= 0.0, 1, -1)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',1,-1)
X=df.iloc[0:100,[0,2]].values

myppn=MyPerceptron(eta=0.1,n_iter=10)
myppn.fit(X,y)
print(myppn.errors_)
print(myppn.w_)

plt.plot(range(1,len(myppn.errors_)+1),myppn.errors_,marker='o')
plt.xlabel('迭代次数')
plt.ylabel('错误分类样本数量')
plt.tight_layout()
plt.show()

