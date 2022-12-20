import xgboost as xgb
import numpy as np

# 1、xgBoost的基本使用
# 2、自定义损失函数的梯度和二阶导
# 3、binary:logistic/logitraw


# 定义f: theta * x
# 类似于定义了逻辑回归
def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h

def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)

if __name__ == "__main__":
    # 读取数据
    data_train = xgb.DMatrix('agaricus.txt.train')
    data_test = xgb.DMatrix('agaricus.txt.test')
    print (data_train)
    print (type(data_train))

    # 设置参数
    param = {'max_depth': 3, 'eta': 1, 'verbosity': 1, 'objective': 'binary:logistic'} # logitraw
    # param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    # 可以显示每一颗树添加后的误差
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 7
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, custom_metric=error_rate)

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print(y_hat)
    print(y)
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print('样本总数：\t', len(y_hat))
    print('错误数目：\t%4d' % error)
    print('错误率：\t%.5f%%' % (100*error_rate))
