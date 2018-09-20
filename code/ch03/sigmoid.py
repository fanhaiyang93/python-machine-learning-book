import matplotlib.pylab as plt
import numpy as np

# 绘制sigmoid曲线

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
z=np.arange(-7,7,0.1)
phi_z=sigmoid(z)
plt.plot(z,phi_z)
plt.axvline(0.0,color='r') # 在x=0处加个竖线
plt.axhline(y=0.5,ls='dotted',color='k')# 在y=0.5处加一个虚线
plt.axhline(y=1,ls='dotted',color='k')# 在y=0.5处加一个虚线
plt.axhline(y=0,ls='dotted',color='k')# 在y=0.5处加一个虚线
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()