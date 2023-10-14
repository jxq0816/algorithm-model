#https://www.jianshu.com/p/3579945a7ea7
# 导入必要的几个包
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 载入数据集,Y的值有0,1,2三种情况，每种特征50个样本
iris = load_iris()
X = iris.data[:, :2]   #获取花卉两列数据集
Y = iris.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

#逻辑回归模型，C=1e5表示目标函数。
lr = LogisticRegression(C=1e5)
lr = lr.fit(X,Y)

print("Logistic Regression模型训练集的准确率：%.3f" %lr.score(x_train, y_train))
print("Logistic Regression模型测试集的准确率：%.3f" %lr.score(x_test, y_test))

#在测试集上进行测试：
from sklearn import metrics
y_hat = lr.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_hat) #错误率，也就是np.average(y_test==y_pred)
print("Logistic Regression模型正确率：%.3f" %accuracy)


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5 # 第0列的范围
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5 # 第1列的范围
h = .02
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h)) # 生成网格采样点

grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
grid_hat = lr.predict(grid_test)                  # 预测分类值
# grid_hat = lr.predict(np.c_[x1.ravel(), x2.ravel()])
grid_hat = grid_hat.reshape(x1.shape)

plt.figure(1, figsize=(6, 5))
# 预测值的显示, 输出为三个颜色区块，分布表示分类的三类区域
plt.pcolormesh(x1, x2, grid_hat,cmap=plt.cm.Paired)

# plt.scatter(X[:, 0], X[:, 1], c=Y,edgecolors='k', cmap=plt.cm.Paired)
plt.scatter(X[:50, 0], X[:50, 1], marker = '*', edgecolors='red', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], marker = '+', edgecolors='k', label='versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1], marker = 'o', edgecolors='k', label='virginica')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend(loc = 2)

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
plt.title("Logistic Regression classification result", fontsize = 15)
plt.xticks(())
plt.yticks(())
plt.grid()

plt.show()