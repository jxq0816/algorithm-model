import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
import matplotlib.pylab as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

features=['devicemodel','resolution']
excelFile = 'devicemodel_resolution_map.csv'
data=pd.DataFrame(pd.read_csv(excelFile))
X=data[['devicemodel','resolution']]
y = data[['tag']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

#LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码。其中包含以下常用方法：
#fit(y) ：fit可看做一本空字典，y可看作要塞到字典中的词。
#fit_transform(y)：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值。
#transform(y) ：将y转变成索引值。

le_x=preprocessing.LabelEncoder()
le_x.fit(np.unique(X_train))
X_train=X_train.apply(le_x.transform)

le_x.fit(np.unique(X_test))
X_test=X_test.apply(le_x.transform)

#XGBoost 使用sklearn wrapper，使用sklearn风格的参数(推荐)
sklearn_model_new = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.5,
    verbosity=1,
    eval_metric='error',
    objective='binary:logistic',
    random_state=1
)
sklearn_model_new.fit(X_train, y_train,eval_set=[(X_test, y_test)])
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
sklearn_model_new2 = xgb.XGBClassifier(
    max_depth=4,
    n_estimators=5,
    verbosity=1,
    eval_metric='error',
    objective='binary:logistic',
    random_state=1
)

#learning_rate用于指定模型迭代的学习率(步长)、默认为0.1；
gsCv2 = GridSearchCV(sklearn_model_new2,{'learning_rate': [0.3,0.5,0.7]})
gsCv2.fit(X_train,y_train)
print(gsCv2.best_score_)
print(gsCv2.best_params_)

#early_stopping_rounds=10,代表在10个迭代内结果没什么改进就停止

sklearn_model_new3 = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.5,
    verbosity=1,
    eval_metric='error',
    objective='binary:logistic',
    n_estimators=10,
    early_stopping_rounds=10
)
sklearn_model_new3.fit(X_train, y_train,eval_set=[(X_test, y_test)])

pred_test_new = sklearn_model_new3.predict(X_test)

dtest = xgb.DMatrix(X_test,y_test)
print (accuracy_score(dtest.get_label(), pred_test_new))