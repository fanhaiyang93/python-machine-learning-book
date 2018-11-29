#!/usr/bin/env python
# coding:utf-8
# @Time    : 2018/11/29 15:19
# @Author  : fanhaiyang
# @File    : AdalineGD.py
# @Software: PyCharm
# @Desc    :实现自适应线性神经元
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plot
class AdalineGD(object):
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

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        训练函数
        :param X: {array_like}, shape={n_samples,n_features}, 训练向量，n_samples是样本的个数，m_features是样本的特征数
        :param y: array_like, shape={n_samples},  目标值
        :return: self: object
        """
        self.w_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            # 每一个权重的更新都使用了全部的数据集
            self.w_[1:] += self.eta * X.T.dot(errors) # 特征矩阵和误差向量的乘积！！
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """
        计算输入值
        :param X: 训练集
        :return: 预测值
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self,X):
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation(X) >=0.0,1,-1)

if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y=df.iloc[0:100,4].values
    y=np.where(y=='Iris-setosa',1,-1)
    X=df.iloc[0:100,[0,2]].values

    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,4))
    ada1=AdalineGD(eta=0.01,n_iter=10).fit(X,y)
    ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(sum_squared_error')

    ada2=AdalineGD(eta=0.0001,n_iter=10).fit(X,y)
    ax[1].plot(range(1,len(ada2.cost_)+1),np.log10(ada2.cost_),marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(sum_squared_error')
    plt.show()
    """
    在学习速率为0.01时，可以发现，学习速率过大，导致随着迭代次数的增加，误差反而在增大，这是因为跳过了全局最优解
    在学习速率为0.0001时，学习速率过小，导致虽然可以最终收敛，但是需要更多的迭代次数
    在实际工作中，经常要对特征值范围进行特征缩放，比如这里对特征数据进行标准化处理。
    标准化：对某个特征值标准化，只需要将其与所有样本的平均值相减，并处以其标准差
    """
    X_std=np.copy(X)
    X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
    X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()

    ada=AdalineGD(n_iter=15,eta=0.01)
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
