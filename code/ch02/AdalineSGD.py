#!/usr/bin/env python
# coding:utf-8
# @Time    : 2018/11/29 15:19
# @Author  : fanhaiyang
# @File    : AdalineGD.py
# @Software: PyCharm
# @Desc    :随机梯度下降
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plot

class AdalineSGD(object):
    '''
    Parameters
    ------------
    eta : float 学习速率，0-1
    n_iter: int  迭代次数
    Attributes
    ------------
    w_: 1d_array 权重向量
    errors_: list 每一次迭代的错误数量
    '''

    def __init__(self, eta=0.01, n_iter=50,shuffle=True,random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized=False
        self.shuffle=shuffle
        if random_state:
            np.random.seed()

    def fit(self, X, y):
        """
        训练函数
        :param X: {array_like}, shape={n_samples,n_features}, 训练向量，n_samples是样本的个数，m_features是样本的特征数
        :param y: array_like, shape={n_samples},  目标值
        :return: self: object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X,y=self._shuffle(X,y)
            cost=[]
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost=sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def net_input(self, X):
        """
        计算输入值
        :param X: 训练集
        :return: 预测值
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def _shuffle(self,X,y):
        r=np.random.permutation(len(y))
        return X[r],y[r]

    def _initialize_weights(self,m):
        self.w_=np.zeros(1+m)
        self.w_initialized=True

    def _update_weights(self,xi,target):
        # 使用单个样本更新权重
        output=self.net_input(xi)
        error=(target-output)
        self.w_[1:]+=self.eta*xi.dot(error)
        self.w_[0]+=self.eta*error
        cost=0.5*error**2
        return cost


    def activation(self,X):
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation(X) >=0.0,1,-1)

if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y=df.iloc[0:100,4].values
    y=np.where(y=='Iris-setosa',1,-1)
    X=df.iloc[0:100,[0,2]].values


    X_std=np.copy(X)
    X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
    X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()

    ada=AdalineSGD(n_iter=15,eta=0.01,random_state=1)
    ada.fit(X_std,y)
    plt.plot(range(1,len(ada.cost_)+1),np.log10(ada.cost_),marker='o')
    plt.xlabel('迭代次数')
    plt.ylabel('误差平方和')
    plt.title('adaline--标准化后')
    plt.show()
    plot.plot_decision_regions(X_std,y,classifier=ada)
    plt.title('adaline--标准化后')
    plt.show()
    """
    可以看出，在标准化后的数据上，使用0.01的学习速率，训练依旧可以收敛。
    """
