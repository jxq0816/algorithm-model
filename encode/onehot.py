# -*- coding: utf-8 -*-

from sklearn.preprocessing import  OneHotEncoder

enc = OneHotEncoder()
enc.fit([[0, 0, 3],
         [1, 1, 0],
         [0, 2, 1],
         [1, 0, 2]])

# 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果
ans = enc.transform([[0, 1, 3]]).toarray()
print(ans)
# 输出 [[ 1.  0.  0.  1.  0.  0.  0.  0.  1.]]