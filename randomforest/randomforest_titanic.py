from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

def randomforest():
	#读取数据
	titan = pd.read_csv('titanic_train.csv')
	#选择特征
	x= titan[['pclass','age','sex']]
	y= titan[['survived']]
	#缺失值处理
	x['age'].fillna(x['age'].mean(),inplace=True)
	#分割数据为训练集和测试集
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
	#特征进行处理
	dict = DictVectorizer(sparse=False)
	x_train = dict.fit_transform(x_train.to_dict(orient='record'))
	x_text = dict.transform(x_text.to_dict(orient='record'))

	#用随机森林进行预测
	rf = RandomForestClassifier(n_estimators =120 ,max_depth=5)
	param = {"n_estimators":[120,200,300,500,800,1200],"max_depth":[5,8,15,25,30]}
	#网格搜索与交叉验证
	gc = GridSearchCV(rf ,param_grid= param,cv=2)
	gc.fit(x_train,y_train)
	print("准确率",gc.score(x_test,y_test))
	print("最优参数",gc.best_params_)

if __name__ == '__main__':
    print('123')
    randomforest()