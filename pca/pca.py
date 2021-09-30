import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#copy表示是否在运行算法时，将原始训练数据复制一份
#n_components指定降维到的维度数目
#svd_solver即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的
#当设置 n_components == 'mle'时，需要和参数svd_solver一起使用，且svd_solver需要选择 'full' 参数；
#即pca = PCA(n_components = 'mle',svd_solver='full')；同时要保证输入数据的样本数多于特征数才可执行成功。
pca = PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
pca.fit(X)

#降维后的各主成分的方差值，方差值越大，则说明越是重要的主成分
print(pca.explained_variance_)
#explained_variance_ratio_计算了每个特征方差贡献率
print(pca.explained_variance_ratio_)
