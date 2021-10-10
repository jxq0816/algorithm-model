from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
#OneHotEncoder(categorical_features='all', dtype=<class 'numpy.float64'>,handle_unknown='error', n_values='auto', sparse=True)
print(enc.categories_)
print(enc.transform([[0, 1, 1]]).toarray())
#array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])