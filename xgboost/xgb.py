import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('/Users/jiangxingqi/AI/xgboost/demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('/Users/jiangxingqi/AI/xgboost/demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
preds_len = len(preds)
for row in range(0,preds_len):
    print(preds[row])