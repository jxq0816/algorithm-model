# coding=utf-8
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

# 导入数据集
df = pd.read_csv("diabetes.csv")
data = df.iloc[:, :8]
target = df.iloc[:, -1]

# 切分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=7)

# xgboost模型初始化设置
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x)
watchlist = [(dtrain, 'train')]

# booster:
params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.75,
          'colsample_bytree': 0.75,
          'min_child_weight': 2,
          'eta': 0.025,
          'seed': 0,
          'nthread': 8,
          'gamma': 0.15,
          'learning_rate': 0.01}

# 建模与预测：50棵树
bst = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)
ypred = bst.predict(dtest)

# 设置阈值、评价指标
y_pred = (ypred >= 0.5) * 1
print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))
print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))
print('F1-score: %.4f' % metrics.f1_score(test_y, y_pred))
print('Accuracy: %.4f' % metrics.accuracy_score(test_y, y_pred))
print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))

ypred = bst.predict(dtest)
print("测试集每个样本的得分\n", ypred)
ypred_leaf = bst.predict(dtest, pred_leaf=True)
print("测试集每棵树所属的节点数\n", ypred_leaf)
ypred_contribs = bst.predict(dtest, pred_contribs=True)
print("特征的重要性\n", ypred_contribs)

xgb.plot_importance(bst, height=0.8, title='影响糖尿病的重要特征', ylabel='特征')
plt.rc('font', family='Arial Unicode MS', size=14)
plt.show()