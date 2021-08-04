from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def add_space(s):
    return ' '.join(list(s))

def jaccard_similarity(s1, s2):
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 获取词表内容
    ret = cv.get_feature_names()
    print(ret)
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator

if __name__ == "__main__":
    s1 = '你在干嘛呢'
    s2 = '你在干什么呢'
    print(jaccard_similarity(s1, s2))
