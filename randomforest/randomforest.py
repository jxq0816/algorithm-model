from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(
        n_samples=1000,# 样本个数
        n_features=4,# 特征个数
        n_informative=2,# 有效特征个数
        n_redundant=0,# 冗余特征个数（有效特征的随机组合）
        random_state=0,
        shuffle=False
)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]]))