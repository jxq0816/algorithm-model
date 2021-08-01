from gensim.test.utils import common_texts,get_tmpfile
from gensim.models import Word2Vec
print(common_texts)
'''
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time'], 
['eps', 'user', 'interface', 'system'], ['system', 'human', 'system', 'eps'], 
['user', 'response', 'time'], ['trees'], ['graph', 'trees'], 
['graph', 'minors', 'trees'], ['graph', 'minors', 'survey']]
'''
model = Word2Vec(common_texts,vector_size=2, window=5, min_count=1, workers=4)
print(model.wv['computer'])
# [-0.41438133 -0.47257847]
