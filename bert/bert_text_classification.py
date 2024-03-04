from transformers import pipeline

classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', top_k=1)

print(classifier("开心"))

print(classifier("happy"))

print(classifier("have a nice day"))

print(classifier("我今天很生气"))

print(classifier("angry"))