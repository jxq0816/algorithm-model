#SnowNLP基于经典机器学习的贝叶斯算法
from snownlp import SnowNLP
print('"这首歌真难听"的情感得分是：',SnowNLP("这首歌真难听").sentiments)
print('"今天天气真好啊"的情感得分是：',SnowNLP("今天天气真好啊").sentiments)