import nltk

nltk.download(
    'wordnet')  # Downloading package wordnet to C:\Users\Administrator\AppData\Roaming\nltk_data...Unzipping corpora\wordnet.zip.

from nltk.corpus import wordnet

word = wordnet.synsets('spectacular')
print(word)
# [Synset('spectacular.n.01'), Synset('dramatic.s.02'), Synset('spectacular.s.02'), Synset('outstanding.s.02')]

print(word[0].definition())
print(word[1].definition())
print(word[2].definition())
print(word[3].definition())
'''
a lavishly produced performance
sensational in appearance or thrilling in effect
characteristic of spectacles or drama
having a quality that thrusts itself into attention
'''