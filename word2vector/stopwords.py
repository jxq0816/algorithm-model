import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')
# Downloading package stopwords to
# C:\Users\Administrator\AppData\Roaming\nltk_data\corpora\stopwords.zip.
# Unzipping the stopwords.zip

"""移除停用词"""
stop_words = stopwords.words("english")

if __name__ == '__main__':
    text = "Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome."

    word_tokens = nltk.tokenize.word_tokenize(text.strip())
    filtered_word = [w for w in word_tokens if not w in stop_words]

    print("word_tokens: ", word_tokens)
    print("filtered_word: ", filtered_word)
    '''
    word_tokens：['Hello', 'Mr.', 'Smith', ',', 'how', 'are', 'you', 'doing', 'today', '?',
     'The', 'weather', 'is', 'great', ',', 'and', 'city', 'is', 'awesome', '.']
    filtered_word：['Hello', 'Mr.', 'Smith', ',', 'today', '?', 'The', 'weather', 'great', ',', 'city', 'awesome', '.']
    '''