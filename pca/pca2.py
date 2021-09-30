import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维
pca2 = PCA(n_components='mle')
pca2.fit(X)
print(pca2.explained_variance_)
print(pca2.explained_variance_ratio_)