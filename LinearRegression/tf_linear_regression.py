#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成线性数据
x = np.linspace(0, 10, 20) + np.random.rand(20)
y = np.linspace(0, 10, 20) + np.random.rand(20)


# 把w,b 定义为变量
w = tf.Variable(np.random.randn() * 0.02)
b = tf.Variable(0.)
print(w.numpy(), b.numpy())  # -0.031422824  0.0


# 定义线性模型
def linear_regression(x):
    return w * x + b

# 定义损失函数
def mean_square_loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# 定义优化器
optimizer = tf.optimizers.SGD()
# 定义优化过程
def run_optimization():
    # 把需要求导的计算过程放入gradient pape中执行,会自动实现求导
    with tf.GradientTape() as g:
        pred = linear_regression(x)
        loss = mean_square_loss(pred, y)
    # 计算梯度
    gradients = g.gradient(loss, [w, b])
    # 更新w, b
    optimizer.apply_gradients(zip(gradients, [w, b]))

# 训练
for step in range(5000):
    run_optimization()   # 持续迭代w, b
    # z展示结果
    if step % 100 == 0:
        pred = linear_regression(x)
        loss = mean_square_loss(pred, y)
        print(f'step:{step}, loss:{loss}, w:{w.numpy()}, b: {b.numpy()}')

linear = LinearRegression()  # 线性回归
linear.fit(x.reshape(-1, 1), y)

plt.scatter(x, y)
x_test = np.linspace(0, 10, 20).reshape(-1, 1)
plt.plot(x_test, linear.coef_ * x_test + linear.intercept_, c='r')  # 画线
plt.plot(x_test, w.numpy() * x_test + b.numpy(), c='g', lw=10, alpha=0.5)  # 画线
plt.show()