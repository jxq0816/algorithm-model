import fasttext

# 训练模型
model = fasttext.train_supervised('train.txt')  # 'train.txt'是你的训练数据文件，需要包含标签和文本

# 测试模型
result = model.test('test.txt')  # 'test.txt'是你的测试数据文件，只需要包含文本
print('Precision:', result[1])
print('Recall:', result[2])

# 对单个文本进行预测
input_text = "This is a sample text"
labels, probabilities = model.predict(input_text, k=1)  # k表示返回的最可能的标签数量
print('Predicted label:', labels[0])
print('Probability:', probabilities[0])