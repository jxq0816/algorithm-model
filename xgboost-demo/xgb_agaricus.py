#基于XGBOST预测毒蘑菇
import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('agaricus.txt.train')
dtest = xgb.DMatrix('agaricus.txt.test')
# specify parameters via map
#max_depth:[default=6]用于指定每个基础模型所包含的最大深度,树高越深，越容易过拟合。
#silent:bool类型参数，是否输出算法运行过程中的日志信息，默认为True。
#eta [default=0.3]:shrinkage参数，用于更新叶子节点权重时，乘以该系数，避免步长过大。参数值越大，越可能无法收敛。把学习率 eta 设置的小一些，小学习率可以使得后面的学习更加仔细。

#objective:定义学习任务及相应的学习目标，“binary:logistic” 表示二分类的逻辑回归问题，输出为概率。
#objective [default=reg:linear]：定义最小化损失函数类型常用参数：
#binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
#multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
#you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
#multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
#设置boosting迭代计算次数
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
preds_len = len(preds)
for row in range(0,preds_len):
    print(preds[row])