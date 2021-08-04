def key_words_ask_method(sentence1,sentence2):
    '''
    因为无论是#1:AVG-W2V 2:AVG-W2V-TFIDF 都需要求得平均值，
    除数：决定整个数据的大小  被除数：影响平均值
    所以 分词的标准很重要，可通过自定义词典、停用词 和 语义分析进行适当处理
    '''
    vec1=sentence_to_vec(sentence1)
    vec2=sentence_to_vec(sentence2)
    #  零向量直接返回
    if(vec1==np.zeros(WORD_VECTOR_DIM)).all()==True or (vec2==np.zeros(WORD_VECTOR_DIM)).all()==True:
        return "不符合相似"

    #  余弦相似度 np.linalg.norm(求范数)（向量的第二范数为传统意义上的向量长度
    dist1 = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    print("score:",dist1)
    if dist1 > 0.92:
        return "两个句子相似"
    else:
        return "两个句子不相似"

if __name__ == "__main__":
    key_words_ask_method("我爱北京","我爱南京")