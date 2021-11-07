import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs

##生产数据
X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

##训练数据
clf = SGDClassifier(loss="hinge", alpha=0.01)
clf.fit(X, Y)

## 绘图
xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)

##生成二维矩阵
X1, X2 = np.meshgrid(xx, yy)
##生产一个与X1相同形状的矩阵
Z = np.empty(X1.shape)
##np.ndenumerate 返回矩阵中每个数的值及其索引
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = clf.decision_function([[x1, x2]]) ##样本到超平面的距离
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
linestyles = ['dashed', 'solid', 'dashed']
colors = 'k'
##绘制等高线：Z分别等于levels
plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
##画数据点
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired,
            edgecolor='black', s=20)
plt.axis('tight')
plt.show()