from nltk.stem import WordNetLemmatizer
#把词的各种派生形式转换为词根
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('leaves'))
# 输出：'leaf'