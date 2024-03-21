import time

import torch
import torch.nn as nn
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset


# 可用设备检测, 有GPU的话将优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 基本的英文分词器
tokenizer = get_tokenizer('basic_english')
# 训练数据加载器
train_iter = AG_NEWS(split="train")
test_iter = AG_NEWS(split="test")


# print('test:')
# train_data = iter(train_iter)
# test_data = iter(test_iter)
# print(next(train_data))
# print(next(test_data))


# 分词生成器
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# 根据训练数据构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
# 设置默认索引，当某个单词不在词汇表 vocab 时（OOV)，返回该单词索引
vocab.set_default_index(vocab["<unk>"])

# 词汇表会将 token 映射到词汇表中的索引上
# print(vocab(["here", "is", "an", "example"]))

# 构建数据加载器 dataloader
# text_pipeline 将一个文本字符串转换为整数 List, List 中每项对应词汇表 vocab 中的单词的索引号
text_pipeline = lambda x: vocab(tokenizer(x))

# label_pipeline 将 label 转换为整数
label_pipeline = lambda x: int(x) - 1


# pipeline example
# print(text_pipeline("hello world! I'am happy"))
# print(label_pipeline("10"))

# 模型
class TextClassificationModule(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        """
            文本分类模型
            description: 类的初始化函数
            :param vocab_size: 整个语料包含的不同词汇总数
            :param embed_dim: 指定词嵌入的维度
            :param num_class: 文本分类的类别总数
        """
        super(TextClassificationModule, self).__init__()
        # 实例化embedding层, sparse=True代表每次对该层求解梯度时, 只更新部分权重
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # 实例化全连接层, 参数分别是embed_dim和num_class
        self.fc = nn.Linear(embed_dim, num_class)
        # 为各层初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化权重函数"""
        # 指定初始权重的取值范围数
        initrange = 0.5
        # 各层的权重参数都是初始化为均匀分布
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        # 偏置初始化为0
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        """
            :param text: 文本数值映射后的结果
            :return: 与类别数尺寸相同的张量, 用以判断文本类别
        """
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# 加载数据集合，转换为张量
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)
# 一个嵌入维度为 64 的模型。词汇大小等于词汇实例的长度。类的数量等于标签的数量，
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModule(vocab_size, emsize, num_class).to(device)

# 训练轮数
EPOCHS = 10
# 学习率
LR = 5
# 训练数据规模
BATCH_SIZE = 64
# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# 调整学习率机制
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

# 划分训练集中5%的数据最为验证集
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

'''使用测试数据集评估模型'''
print('Checking the results of test dataset.')
accu_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test))

# 测试随机新闻
# 使用迄今为止最好的模型并测试高尔夫新闻。
ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

model = model.to("cpu")

print("This is a %s news" % ag_news_label[predict(ex_text_str, text_pipeline)])