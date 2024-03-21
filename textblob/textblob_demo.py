from textblob.classifiers import NaiveBayesClassifier

train = [
    ('宝马', '汽车'),
    ('肖战', '明星')
]

nb_model = NaiveBayesClassifier(train)

dev_sen = "奔驰"
print(nb_model.classify(dev_sen))
