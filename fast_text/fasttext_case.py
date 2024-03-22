import fasttext
fasttext.FastText.eprint = lambda x: None


def load_model():
    return fasttext.load_model("fasttext.bin")


model = load_model()
rs = model.predict("it is so good!")
print(rs)

rs = model.predict("it is so bad!")
print(rs)