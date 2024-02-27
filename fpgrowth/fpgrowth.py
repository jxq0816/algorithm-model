import pyfpgrowth

# 输入数据：事务列表
transactions = [
    ['牛奶', '面包', '黄油'],
    ['牛奶', '面包'],
    ['啤酒', '面包']
]

# 设置支持度阈值，这里我们使用2作为最小支持度
min_support = 2

# 使用pyfpgrowth找出频繁项集和它们的支持度
patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)

# 输出结果
print("频繁项集及其支持度：", patterns)