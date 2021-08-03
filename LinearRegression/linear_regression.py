# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression

x = [
    [80, 86],
    [82, 80],
    [85, 78],
    [90, 90],
    [86, 82],
    [82, 90],
    [78, 80]
]
y = [84.2, 80.6, 80.1, 90, 83.2,87.6,79.4]

estimator = LinearRegression()

estimator.fit(x,y)
#coef参数w1到wn
_coef = estimator.coef_
print(_coef)
#intercept_为w0
_intercept = estimator.intercept_
print(_intercept)
