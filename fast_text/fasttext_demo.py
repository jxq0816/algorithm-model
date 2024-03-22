import os
import fasttext
from sklearn.metrics import classification_report


def save_model(model):
    model.save_model("fasttext.bin")


def load_model():
    return fasttext.load_model("fasttext.bin")


def evaluate_model(test_file):
    inputs = []
    y_true = []
    for line in open(test_file,encoding='utf-8'):
        if line.strip() == "":
            continue
        tmp = line.strip().split("\t")
        label = tmp[0]
        text = "\t".join(tmp[1:])
        y_true.append(label)
        inputs.append(text)
    model = load_model()
    y_pred = model.predict(inputs)[0]
    report = classification_report(y_true, y_pred)
    return report


if __name__ == "__main__":

    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)
    train_filepath = os.path.join(parent_path,'0_data_set','imdb','train.txt')
    test_filepath = os.path.join(parent_path, '0_data_set', 'imdb', 'test.txt')
    #model = fasttext.train_supervised(train_filepath, epoch=50, lr=0.05, dim=300, maxn=128)
    #save_model(model)
    report = evaluate_model(test_filepath)
    print(report)
    with open("fasttext_exp.log", "w+") as fp:
        fp.write(report)