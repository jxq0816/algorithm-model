import numpy as np
from sklearn import linear_model
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
clf = linear_model.SGDRegressor()
clf.fit(X, y)

#coef参数w1到wn
_coef = clf.coef_
print(_coef)
#intercept_为w0
_intercept = clf.intercept_
print(_intercept)