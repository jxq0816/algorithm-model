# -*- coding: utf-8 -*-

from sklearn.preprocessing import  OneHotEncoder

enc = OneHotEncoder(sparse = False)
ans = enc.fit_transform([[0, 0, 3],
                         [1, 1, 0],
                         [0, 2, 1],
                         [1, 0, 2]])

print(ans)
# 输出 [[ 1.  0.  1. ...,  0.  0.  1.]
#      [ 0.  1.  0. ...,  0.  0.  0.]
#      [ 1.  0.  0. ...,  1.  0.  0.]
#      [ 0.  1.  1. ...,  0.  1.  0.]]
