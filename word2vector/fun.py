#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-9-28 22:21
# @Author  : Manu
# @Site    :
# @File    : python_base.py
# @Software: PyCharm

from __future__ import division
import nltk
import matplotlib
from nltk.book import *
from nltk.util import bigrams

# 单词搜索
print('单词搜索')
text1.concordance('boy')
text2.concordance('friends')

# 相似词搜索
print('相似词搜索')
#print(text3.similar('time'))

# 共同上下文搜索
print('共同上下文搜索')
text2.common_contexts(['monstrous', 'very'])

# 词汇分布表
print('词汇分布表')
#text4.dispersion_plot(['citizens', 'American', 'freedom', 'duties'])

# 词汇计数
print('词汇计数')
print(len(text5))
sorted(set(text5))
print(len(set(text5)))

# 重复词密度
print('重复词密度')
print(len(text8) / len(set(text8)))

# 关键词密度
print('关键词密度')
print(text9.count('girl'))
print(text9.count('girl') * 100 / len(text9))

# 频率分布
fdist = FreqDist(text1)

vocabulary = fdist.keys()
#for i in vocabulary:
#    print(i)

# 高频前20
fdist.plot(20, cumulative=True)

# 低频词
print('低频词：')
#print(fdist.hapaxes())

# 词语搭配
print('词语搭配')
words = list(bigrams(['louder', 'words', 'speak']))
print(words)