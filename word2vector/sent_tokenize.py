from nltk.tokenize import sent_tokenize

text = """Hello Mr. Smith, how are you doing today? The weather is great, and 
city is awesome.The sky is pinkish-blue. You shouldn't eat cardboard"""

tokenized_text = sent_tokenize(text)

print(tokenized_text)
'''
结果：
  ['Hello Mr. Smith, how are you doing today?', 
   'The weather is great, and city is awesome.The sky is pinkish-blue.', 
   "You shouldn't eat cardboard"]
'''