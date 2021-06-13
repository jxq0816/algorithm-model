# -*- coding: utf-8 -*-

from sklearn.preprocessing import  OneHotEncoder

enc = OneHotEncoder(n_values = [2, 3, 4])
enc.fit([[0, 0, 3],
         [1, 1, 0]])

ans = enc.transform([[0, 2, 3]]).toarray()
print(ans) # 输出 [[ 1.  0.  0.  0.  1.  0.  0.  0.  1.]]