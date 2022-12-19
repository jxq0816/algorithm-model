import string

"""移除标点符号"""
if __name__ == '__main__':
    # 方式一
    # s = 'abc.'
    text_list = "Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome."
    text_list = text_list.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))  # abc
    print("s: ", text_list)
    #s:  Hello Mr  Smith  how are you doing today  The weather is great  and city is awesome

    # 方式二
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    text_list = [word for word in text_list if word not in english_punctuations]
    print("text: ", text_list)
    #text:  ['H', 'e', 'l', 'l', 'o', ' ', 'M', 'r', ' ', ' ', 'S', 'm', 'i', 't', 'h', ' ', ' ',
    # 'h', 'o', 'w', ' ', 'a', 'r', 'e', ' ', 'y', 'o', 'u', ' ', 'd', 'o', 'i', 'n', 'g', ' ', 't', 'o', 'd', 'a', 'y', '
    # ', ' ', 'T', 'h', 'e', ' ', 'w', 'e', 'a', 't', 'h', 'e', 'r', ' ', 'i', 's', ' ', 'g', 'r', 'e', 'a', 't', '
    # ', ' ', 'a', 'n', 'd', ' ', 'c', 'i', 't', 'y', ' ', 'i', 's', ' ', 'a', 'w', 'e', 's', 'o', 'm', 'e', ' ']