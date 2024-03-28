# 导入所需的库
import numpy as np  # NumPy用于数值运算
import matplotlib.pyplot as plt  # Matplotlib用于绘图
from sklearn.cluster import KMeans  # 从sklearn库导入KMeans用于聚类
from sklearn.datasets import make_blobs  # make_blobs用于生成合成数据集
from sklearn.metrics import silhouette_score  # silhouette_score用于计算轮廓分数
from scipy.spatial.distance import cdist  # cdist用于计算点之间的距离

# 生成合成数据集，可以用自己的数据集替换这部分
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# 初始化一个列表用于存储不同k值的SSE（误差平方和）值
sse = []

# 初始化一个列表用于存储不同k值的轮廓分数
silhouette_scores = []

# 定义一个k值的范围进行尝试
k_values = range(2, 10)

# 对于k值范围内的每一个k，执行以下操作
for k in k_values:
    # 使用k个聚类中心初始化KMeans模型
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)  # 对数据集X进行拟合

    # 计算并添加当前k值的SSE
    sse.append(kmeans.inertia_)

    # 计算并添加当前k值的轮廓分数
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# 绘制不同k值的SSE值图
plt.figure(figsize=(12, 6))  # 设置图形大小
plt.subplot(1, 2, 1)  # 定义子图布局
plt.plot(k_values, sse, 'bo-')  # 绘制SSE值曲线图
plt.title('Elbow Method For Optimal k')  # 添加标题
plt.xlabel('Number of clusters (k)')  # x轴标签
plt.ylabel('SSE')  # y轴标签

# 绘制不同k值的轮廓分数图
plt.subplot(1, 2, 2)  # 定义第二个子图布局
plt.plot(k_values, silhouette_scores, 'go-')  # 绘制轮廓分数曲线图
plt.title('Silhouette Method For Optimal k')  # 添加标题
plt.xlabel('Number of clusters (k)')  # x轴标签
plt.ylabel('Silhouette Score')  # y轴标签

plt.tight_layout()  # 调整子图布局
plt.show()  # 展示图形