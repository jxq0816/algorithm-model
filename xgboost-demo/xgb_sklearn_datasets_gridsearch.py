import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

#dtrain = xgb.DMatrix(X_train,y_train)
#dtest = xgb.DMatrix(X_test,y_test)
#param = {'max_depth':5, 'eta':0.5, 'verbosity':1, 'objective':'binary:logistic'}
#XGBoost 使用sklearn wrapper，仍然使用原始API的参数
#sklearn_model_raw = xgb.XGBClassifier(**param)
#sklearn_model_raw.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",eval_set=[(X_test, y_test)])
#XGBoost 使用sklearn wrapper，使用sklearn风格的参数(推荐)
sklearn_model_new = xgb.XGBClassifier(max_depth=5,learning_rate= 0.5, verbosity=1, objective='binary:logistic',random_state=1)
sklearn_model_new.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",eval_set=[(X_test, y_test)])
#使用sklearn网格搜索调参
#n_estimators则是非常重要的要调的参数，它关系到我们XGBoost模型的复杂度，因为它代表了我们决策树弱学习器的个数。
#n_estimators太小，容易欠拟合，n_estimators太大，模型会过于复杂，一般需要调参选择一个适中的数值。
#一般固定步长，先调好框架参数n_estimators，再调弱学习器参数max_depth，min_child_weight,gamma等，
#接着调正则化相关参数subsample，colsample_byXXX, reg_alpha以及reg_lambda,最后固定前面调好的参数，来调步长learning_rate
gsCv = GridSearchCV(sklearn_model_new,{'max_depth': [4,5,6],'n_estimators': [5,10,20]})
gsCv.fit(X_train,y_train)
print(gsCv.best_score_)
print(gsCv.best_params_)

#n_estimators用于指定基础模型的数量、默认为100个。
sklearn_model_new2 = xgb.XGBClassifier(max_depth=4,n_estimators=10,verbosity=1, objective='binary:logistic',random_state=1)

#learning_rate用于指定模型迭代的学习率(步长)、默认为0.1；
gsCv2 = GridSearchCV(sklearn_model_new2,{'learning_rate ': [0.3,0.5,0.7]})
gsCv2.fit(X_train,y_train)

print(gsCv2.best_score_)
print(gsCv2.best_params_)

#sklearn_model_new2 = xgb.XGBClassifier(max_depth=4,learning_rate= 0.3, verbosity=1, objective='binary:logistic',n_estimators=10)
#sklearn_model_new2.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",eval_set=[(X_test, y_test)])

#pred_test_new = sklearn_model_new2.predict(X_test)
#print (accuracy_score(dtest.get_label(), pred_test_new))