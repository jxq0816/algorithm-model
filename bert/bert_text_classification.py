from transformers import pipeline

classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', top_k=1)

print('好开心',classifier("好开心"))

print('He is a interesting man',classifier("He is a interesting man"))

print('have a nice day',classifier("have a nice day"))

print('我今天很生气',classifier("我今天很生气"))

print('i am sad',classifier("angry"))