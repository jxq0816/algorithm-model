#https://zhuanlan.zhihu.com/p/657231641
from datasketch import MinHash, MinHashLSH
# 创建一个 MinHash 对象
def create_minhash(data):
    minhash = MinHash(num_perm=128)  # num_perm 是哈希函数的数量，可以根据需要调整
    for d in data:
        minhash.update(d.encode('utf8'))
    return minhash
# 创建一些示例数据（中文长句子）
sentences = [
    "今天天气很好，阳光明媚，适合出门散步。",
    "我喜欢读书，尤其是科幻小说。",
    "这个城市的夜景非常漂亮，尤其是灯光璀璨的CBD区。",
    "我的家乡是一个美丽的小镇，四季分明，景色宜人。",
    "学习新知识让我感到充实和快乐。",
    "我喜欢健身，每天都会去健身房锻炼。",
    "这家餐厅的菜品非常美味，尤其是他们的特色菜。",
    "我喜欢旅行，尤其喜欢去一些自然风光优美的地方。",
    "听音乐是我放松心情的最爱之一。",
    "看电影是我周末最喜欢做的事情之一，我喜欢各种类型的电影。"
]

# 创建 MinHash 对象并插入到 LSH 中
lsh = MinHashLSH(threshold=0.5, num_perm=128)  # threshold 是相似度阈值，可以根据需要调整

for idx, sentence in enumerate(sentences):
    minhash = create_minhash(list(sentence))
    lsh.insert(idx, minhash)
# 查找相似的集合
query_minhash = create_minhash(list('听音乐是我放松心情的最爱'))
results = lsh.query(query_minhash)
# 输出相似度分数
for result in results:
    minhash = create_minhash(list(sentences[result]))
    jaccard_similarity = query_minhash.jaccard(minhash)
    print(f"与 sentence 相似的句子 {result} 的相似度分数为: {jaccard_similarity}")
# output