# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error


def model():
    """线性回归模型"""
    # 获取数据
    data = load_boston()
    # 数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    # 特征工程
    stand = StandardScaler()
    x_train = stand.fit_transform(x_train)
    x_test = stand.fit_transform(x_test)
    # 建立模型
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    # 模型评估

    pre = estimator.predict(x_test)
    print("预测值：", pre)

    score = estimator.score(x_test, y_test)
    print("准确率：", score)

    # 均方误差
    ret = mean_squared_error(y_test, pre)
    print("均方误差是", ret)

if __name__ == '__main__':
    model()