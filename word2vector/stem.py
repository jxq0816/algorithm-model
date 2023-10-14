# 基于Porter词干提取算法
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
print(porter_stemmer.stem('maximum'))

# 基于Lancaster 词干提取算法
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('maximum'))

# 基于Snowball 词干提取算法
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
print(snowball_stemmer.stem('maximum'))

from nltk.stem.wordnet import WordNetLemmatizer  # from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()  # 词形还原
from nltk.stem.porter import PorterStemmer  # from nltk.stem import PorterStemmer
stem = PorterStemmer()  # 词干提取
word = "flying"
print("Lemmatized Word:", lem.lemmatize(word, "v"))
print("Stemmed Word:", stem.stem(word))
'''
Lemmatized Word: fly
Stemmed Word: fli
'''