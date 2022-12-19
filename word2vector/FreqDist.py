import nltk
all_words = nltk.FreqDist(w.lower()  for  w  in  nltk.word_tokenize( "I'm foolish foolish man" ))
print (all_words.keys())
all_words.plot()