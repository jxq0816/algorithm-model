import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

corpus = ['The sky is blue and beautiful.',
          'Love this blue and beautiful sky!',
          'The quick brown fox jumps over the lazy dog.',
          'The brown fox is quick and the blue dog is lazy!',
          'The sky is very blue and the sky is very beautiful today',
          'The dog is lazy but the brown fox is quick!'
]

labels = ['weather', 'weather', 'animals', 'animals', 'weather', 'animals']

# 第一步：进行DataFrame化操作

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 'category': labels})

# 第二步：进行分词和停用词的去除
import nltk

stopwords = nltk.corpus.stopwords.words('english')
wps = nltk.WordPunctTokenizer()
def Normalize_corpus(doc):

    tokens = re.findall(r'[a-zA-Z0-9]+', doc.lower())
    doc = [token for token in tokens if token not in stopwords]
    doc = ' '.join(doc)
    return doc

# 第三步：向量化函数，调用函数进行分词和停用词的去除
Normalize_corpus = np.vectorize(Normalize_corpus)
corpus_array = Normalize_corpus(corpus)

# 第四步：对单个词计算word2vec特征向量
from gensim.models import word2vec
corpus_token = [wps.tokenize(corpus) for corpus in corpus_array]
print(corpus_token)
# 特征的维度
feature_size = 10
# 最小的统计个数，小于这个数就不进行统计
min_count = 1
# 滑动窗口
window = 10
# 对出现次数频繁的词进行随机下采样操作
sample = 1e-3
model = word2vec.Word2Vec(corpus_token, size=feature_size, min_count=min_count, window=window, sample=sample)
print(model.wv.index2word)

# 第五步：对每一个corpus做平均的word2vec特征向量
def word2vec_corpus(corpuses, num_size=10):

    corpus_tokens = [wps.tokenize(corpus) for corpus in corpuses]
    model = word2vec.Word2Vec(corpus_tokens, size=num_size, min_count=min_count, window=window, sample=sample)
    vocabulary = model.wv.index2word
    score_list = []
    for corpus_token in corpus_tokens:
        count_time = 0
        score_array = np.zeros([10])
        for word in corpus_token:
            if word in vocabulary:
                count_time += 1
                score_array += model.wv[word]
        score_array = score_array / count_time
        score_list.append(list(score_array))

    return score_list

print(np.shape(word2vec_corpus(corpus_array, num_size=10)))