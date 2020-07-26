# encoding: utf-8
# 基于LSTM的seq2sql机器翻译， pytorch实现, 德语到英语的翻译
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time

# 设置随机种子
SEED = 20
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 加载模型
spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

# 使用spacy创建分词器（tokenizers）， 作者发现颠倒源语言的输入的顺序可以取得不错的翻译效果


def tokenize_de(text):
    # 德语的分词， 并且颠倒顺序
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    # 英语分词， 不颠倒顺序
    return [tok.text for tok in spacy_en.tokenizer(text)]


# 创建SRC和TRG两个Field对象，tokenize为我们刚才定义的分词器函数，在每句话的开头加入字符SOS，结尾加入字符EOS，将所有单词转换为小写。
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

# 加载训练集、验证集和测试集，生成dataset类。使用torchtext自带的Multi30k数据集，这是一个包含约30000个平行的英语、德语和法语句子的数据集，每个句子包含约12个单词。
# splits方法可以同时加载训练集，验证集和测试集，参数exts指定使用哪种语言作为源语言和目标语言，fileds指定定义好的Field类
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
print(len(train_data.examples))
print(vars(train_data.examples[0]))

# 生成字典， 设置最小词频为3
SRC.build_vocab(train_data, min_freq=3)
TRG.build_vocab(train_data, min_freq=3)
print(len(SRC.vocab))
print(len(TRG.vocab))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置batch_size
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)
# batch = next(iter(train_iterator))
# print(batch)

# 创建seq2sql模型， [Encoder, Decoder, Seq2seq]
# Encoder层， 论文4层单向LSTM， 缩减为两层


class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # 生成稀疏的词向量
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 创建lstm网络
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, dropout=dropout)
        # dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # shape: src(sent_len, batch_size)
        embedded = self.dropout(self.embedding(src))
        # shape: embedded(sent_len, batch_size, emb_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (sent_len, batch_size, hid_dim)
        # hidden: (n_layers, batch_size, hid_dim)
        # cell: (n_layers, batch_size, hid_dim)
        return hidden, cell


class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        # LSTM之后全连接层预测输出结果
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_input, hidden, cell):
        # lstm_input:(batch_size) -> lstm_input:(1, batch_size)
        lstm_input = lstm_input.unsqueeze(0)
        # embedded: (1, batch_size, emb_dim)
        embedded = self.dropout(self.embedding(lstm_input))
        # hidden:(n_layers, batch size, hid_dim)
        # cell:(n_layers, batch size, hid_dim)
        # output(1, batch_size, hid_dim)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # prediction: (batch_size, output_dim)
        predication = self.out(output.squeeze(0))

        return predication, hidden, cell


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, devices):
        super(Seq2Seq, self).__init__()
        # 在这里encoder和decoder的hidden_dim和layers要一致，便于连接
        self.encoder = encoder
        self.decoder = decoder
        self.device = devices

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (sent_len, batch size)
        # trg: (sent_len, batch size)
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # 创建outputs张量存储Decoder的输出
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        # 输入到decoder网络的第一个字符是<sos>(句子开始标志)
        decoder_input = trg[0, :]

        for t in range(1, max_len):
            # 循环更新hidden， cell，两层
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = (trg[t] if teacher_force else top1)

        return outputs


def init_weight(m):
    # 初始化权重
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def train(torch_model, iterator, optimizer, criterion, clip):
    torch_model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()

        outputs = torch_model(src, trg)
        # trg: (sent_len, batch size) -> (sent_len-1) * batch size)
        # output: (sent_len, batch_size, output_dim) -> ((sent_len-1) * batch_size, output_dim))
        outputs = outputs[1:].view(-1, outputs.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(outputs, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(torch_model, iterator, criterion):

    torch_model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            outputs = torch_model(src, trg, 0)
            outputs = outputs[1:].view(-1, outputs.shape[-1])
            trg = trg[1:].view(-1)

            try:
                loss = criterion(outputs, trg)
            except ValueError:
                print(111)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # 训练控制
    is_train = True

    # 模型训练
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    N_EPOCHS = 10
    CLIP = 1

    # 初始化为正无穷
    best_valid_loss = float('inf')

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(encoder=enc, decoder=dec, devices=device).to(device=device)

    if is_train:
        # 初始化权重参数
        model.apply(init_weight)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

        # padding相同长度， 并补齐
        PAD_IDX = TRG.vocab.stoi['<pad>']

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 't1-model.pt')
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    else:
        # padding相同长度， 并补齐
        PAD_IDX = TRG.vocab.stoi['<pad>']

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        model.load_state_dict(torch.load('t1-model.pt'))
        test_loss = evaluate(model, test_iterator, criterion)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

