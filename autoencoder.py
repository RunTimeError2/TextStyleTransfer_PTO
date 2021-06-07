# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as functional
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import Config
from dataset import Vocabulary, DualNovelDataSet


config = Config()


class SeqAutoEncoder(object):
    """
    序列自编码器
    用于计算语句的语义差别
    """

    def __init__(self):
        """
        初始化类
        步骤包括设置数据集、初始化模型和优化器、设定参数
        """
        # 设置数据集
        self.vocabulary = Vocabulary(config.vocab_file)
        self.pad = self.vocabulary.word2id['<pad>']
        self.go = self.vocabulary.word2id['<go>']
        self.eos = self.vocabulary.word2id['<eos>']
        self.unk = self.vocabulary.word2id['<unk>']

        self.train_set = DualNovelDataSet()
        self.test_set = DualNovelDataSet()

        # 初始化模型
        self.encoder = RNNEncoder(config.encoder_num_layers, config.encoder_bidirectional)
        self.decoder = RNNDecoder(config.decoder_num_layers, config.decoder_bidirectional)

        self.trainable_variables = []
        for k, v in self.encoder.state_dict(keep_vars=True).items():
            if v.requires_grad:
                self.trainable_variables.append(v)
        for k, v in self.decoder.state_dict(keep_vars=True).items():
            if v.requires_grad:
                self.trainable_variables.append(v)

        # 设定优化器和参数
        self.learning_rate = config.ae_learning_rate
        self.beta1 = config.ae_beta1
        self.beta2 = config.ae_beta2
        self.optimizer = Adam(self.trainable_variables, self.learning_rate, (self.beta1, self.beta2))

        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.batch_size = config.ae_batch_size
        self.epochs = config.ae_epochs
        self.num_workers = config.ae_num_workers

    def set_training(self, train_mode):
        """设定训练/测试模式

        Args:
            train_mode: 布尔型，是否是训练模式
        """
        self.encoder.train(mode=train_mode)
        self.decoder.train(mode=train_mode)

    def train(self, verbose=False, graph=False):
        """训练自编码器

        Args:
            verbose: 是否输出提示信息，即每个Epoch结束后输出该代的损失
            graph: 训练结束后是否显示损失变化曲线图
        """
        loss_list = []

        for epoch in range(self.epochs):
            epoch_loss = self.run_epoch(test=False)
            loss_list += epoch_loss
            if verbose:
                print('\n[TRAIN] Epoch {}, mean loss {}'.format(epoch, np.mean(epoch_loss)))

        if graph:
            plt.figure()
            plt.plot([x for x in range(len(loss_list))], loss_list)
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.grid()
            plt.title('Training loss')
            plt.show()

    def run_epoch(self, test=False):
        """运行一个epoch，可以指定训练/测试模式

        Args:
            test: 布尔型，是否是测试模式

        Returns:
            test = True:
                mean_loss: 平均损失含函数值
            test = False:
                loss_list: 训练过程中的损失函数，是一个列表
        """
        loss_list = []

        if test:
            loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.encoder.train(mode=True)
            self.decoder.train(mode=True)
        else:
            loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.encoder.train(mode=False)
            self.encoder.train(mode=False)

        with tqdm(loader) as pbar:
            for data in pbar:
                sentences, labels = self.preprocess_data(data)
                batch_size = sentences.shape[0]

                encoder_state = self.encoder.init_hidden(batch_size)
                encoder_output, encoder_hidden = self.encoder(sentences, encoder_state)

                decoder_input = self.go * torch.ones((batch_size, config.max_sentence_length, config.embedding_dim))
                decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_hidden)
                loss = self.criterion(decoder_output, sentences.reshape(-1))

                if not test:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                loss_list.append(loss.item())

        return np.mean(loss_list) if test else loss_list

    def mean_difference(self, sentences_0, sentences_1):
        """计算两组语句的平均语义差别
        将语句进行编码，语义差别定义为其编码结果的均方差

        Args:
            sentences_0: 形状为[batch_size, max_seq_len]的LongTensor，表示语句
            sentences_1: 同上

        Returns:
            mean_difference: 平均语义差别
        """
        with torch.no_grad():
            assert sentences_0.shape[0] == sentences_1.shape[0]
            batch_size = sentences_0.shape[0]

            encoder_state = self.encoder.init_hidden(batch_size)

            _, hidden_0 = self.encoder(sentences_0, encoder_state)
            _, hidden_1 = self.encoder(sentences_1, encoder_state)

            return self.mse_loss(hidden_0, hidden_1)

    def save_model(self):
        """
        将模型保存到指定路径
        """
        torch.save(self.encoder.state_dict(), config.encoder_model_path)
        torch.save(self.decoder.state_dict(), config.decoder_model_path)

    def load_model(self):
        """
        从指定路径的文件读取模型参数
        """
        self.encoder.load_state_dict(torch.load(config.encoder_model_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(config.decoder_model_path, map_location=lambda storage, loc: storage))

    def preprocess_data(self, data):
        """预处理数据
        这里的数据是来自DualNovelDataSet的数据样本，同时包括风格为0和1的数据

        Args:
            data: DataLoader给出的每一个数据样本，格式见本函数第一行

        Returns:
            sentences: 形状为[batch_size * 2, max_len]的Tensor，代表语句数据
            label: 形状为[batch_size * 2]的Tensor，代表标签
        """
        (bare_0, go_0, eos_0, len_0), (bare_1, go_1, eos_1, len_1) = data
        batch_size = bare_0.shape[0]

        label_0 = torch.zeros(batch_size)
        label_1 = torch.ones(batch_size)

        sentences = torch.cat([bare_0, bare_1], dim=0)
        label = torch.cat([label_0, label_1], dim=0)

        if config.gpu:
            sentences = sentences.cuda()
            label = label.cuda()

        return sentences, label


class RNNEncoder(nn.Module):
    """
    基于RNN的序列编码器
    结构包含一个Embedding层和一个多层GRU模块
    """

    def __init__(self, num_layers, bidirectional):
        """初始化编码器

        Args:
            num_layers: GRU网络的层数
            bidirectional: 布尔型，是否使用双向网络
        """
        super(RNNEncoder, self).__init__()

        self.vocabulary = Vocabulary(config.vocab_file)
        self.embedding_dim = config.embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(self.vocabulary.vocab_size, self.embedding_dim)
        if config.gpu:
            self.embedding = self.embedding.cuda()

        self.gru = nn.GRU(
            self.embedding_dim, self.embedding_dim, num_layers, batch_first=True, bidirectional=bidirectional
        )
        if config.gpu:
            self.gru = self.gru.cuda()

    def forward(self, inputs, hidden):
        """前向传播步骤

        Args:
            inputs: 形状为[batch_size, max_seq_len]的LongTensor，表示网络输入
            hidden: 形状为[num_layers * directions, batch_size, embedding_dim]的Tensor，表示隐藏状态

        Returns:
            output: 网络输出，形状为[batch_size, max_seq_len, embedding_dim]的Tensor
            hidden: 网络输出的状态，形状同输入的hidden
        """
        # embedded shape: [batch_size, max_seq_len, embedding_dim]
        embedded = self.embedding(inputs)
        # output shape: [batch_size, max_seq_len, embedding_dim]
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        """取得用于初始化的全零状态

        Args:
            batch_size: 一批数据的大小，和初始状态的维度有关

        Returns:
            state: 可用于初始化的隐藏状态，是形状为[num_layers * directions, batch_size, embedding_dim]的Tensor
        """
        state = torch.zeros((self.num_layers * self.directions, batch_size, self.embedding_dim))
        if config.gpu:
            state = state.cuda()
        return state


class RNNDecoder(nn.Module):
    """
    基于RNN的解码器模块
    结构包含一个多层GRU模块和一个全连接层
    解码器不需要初始化状态，因为初始状态来自编码器
    """

    def __init__(self, num_layers, bidirectional):
        """初始化解码器

        Args:
            num_layers: GRU网络的层数
            bidirectional: 布尔型，是否使用双向网络
        """
        super(RNNDecoder, self).__init__()

        self.vocabulary = Vocabulary(config.vocab_file)
        self.embedding_dim = config.embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(self.embedding_dim, self.embedding_dim)
        if config.gpu:
            self.embedding = self.embedding.cuda()

        self.gru = nn.GRU(
            self.embedding_dim, self.embedding_dim, num_layers, batch_first=True, bidirectional=bidirectional
        )
        if config.gpu:
            self.gru = self.gru.cuda()

        self.dense = nn.Linear(self.embedding_dim * self.directions, self.vocabulary.vocab_size)
        if config.gpu:
            self.dense = self.dense.cuda()

    def forward(self, inputs, hidden):
        """前向传播步骤

        Args:
            inputs: 形状为[batch_size, max_seq_len]的LongTensor，表示网络输入，一般用起始符号填充
            hidden: 形状为[num_layers * directions, batch_size, embedding_dim]的Tensor，表示隐藏状态，来自编码器

        Returns:
            output_logits: 形状为[batch_size * max_seq_len, vocab_size]的Tensor，表示解码后得到的语句
                           中各个单词出现概率的评分（one-hot编码），前两个维度被flatten了
            output_probs: output_logits经过softmax的结果，表示各单词出现的概率
        """
        output = inputs  # shape: [batch_size, max_seq_len, embedding_dim]
        output, hidden = self.gru(output, hidden)
        output = output.reshape(output.size(0) * output.size(1), output.size(2))
        # output_logits shape: [batch_size * max_seq_len, vocab_size]
        output_logits = self.dense(output)
        output_probs = functional.softmax(output_logits, dim=1)

        return output_logits, output_probs


def train_autoencoder():
    model = SeqAutoEncoder()
    model.train(verbose=True, graph=True)
    model.save_model()

    loader = DataLoader(model.train_set, batch_size=16, shuffle=True)
    for data in loader:
        (sen_0, _, _, _), (sen_1, _, _, _) = data
        print(model.mean_difference(sen_0, sen_1))
        break


if __name__ == '__main__':
    train_autoencoder()
