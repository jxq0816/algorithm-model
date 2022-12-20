from sklearn.preprocessing import LabelEncoder,OneHotEncoder
enc = OneHotEncoder()
lb = LabelEncoder()
tmp = lb.fit_transform([123,456,789])
print(tmp)
#reshape(-1,1)转化成1列
#将LabelEncoder的结果作为OneHotEncoder特征输入
tmp_reshape=tmp.reshape(-1,1)
print(tmp_reshape)
enc.fit(tmp_reshape)
x_train = enc.transform(lb.transform([456,789,123]).reshape(-1, 1)).toarray()
#输出特征[123,789]的OneHotEncoder的编码结果
print(x_train)