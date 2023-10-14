import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 这个DMatrix我们不需要关心里面的细节，使用我们的训练集X和y初始化即可。
# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

#上面的代码中，我们随机初始化了一个二分类的数据集，然后分成了训练集和验证集。
#使用训练集和验证集分别初始化了一个DMatrix，有了DMatrix，就可以做训练和预测了
#原生XGBoost需要先把数据集按输入特征部分，输出部分分开，然后放到一个DMatrix数据结构里面，
dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test,y_test)
param = {'max_depth':5, 'eta':0.5, 'verbosity':1, 'objective':'binary:logistic'}
raw_model = xgb.train(param, dtrain, num_boost_round=20)


from sklearn.metrics import accuracy_score
pred_train_raw = raw_model.predict(dtrain)
for i in range(len(pred_train_raw)):
    if pred_train_raw[i] > 0.5:
         pred_train_raw[i]=1
    else:
        pred_train_raw[i]=0
print (accuracy_score(dtrain.get_label(), pred_train_raw))

#训练集的准确率我这里输出是0.9664。再看看验证集的表现：

pred_test_raw = raw_model.predict(dtest)
for i in range(len(pred_test_raw)):
    if pred_test_raw[i] > 0.5:
         pred_test_raw[i]=1
    else:
        pred_test_raw[i]=0
print (accuracy_score(dtest.get_label(), pred_test_raw))

#验证集的准确率我这里的输出是0.9408，已经很高了。