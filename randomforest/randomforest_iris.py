from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

iris = load_iris()   # 这里是sklearn中自带的一部分数据
df = pd.DataFrame(iris.data, columns=iris.feature_names) # 格式化数据
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)  ## 新接口 数据
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:4]
clf = RandomForestClassifier(n_jobs=2)
y, _ = pd.factorize(train['species'])
clf.fit(train[features], y)  # 用train来训练样本

test_pred=clf.predict(test[features])   #用测试数据来做预测
preds = iris.target_names[test_pred]
pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])