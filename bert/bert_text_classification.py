from transformers import pipeline

classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', top_k=1)

prediction = classifier("开心")

print(prediction)

prediction = classifier("生气")

print(prediction)