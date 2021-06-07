# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import numpy as np
from config import Config
from dataset import Vocabulary, sample_2d


config = Config()


class Delete(object):
    """
    执行删除操作的类
    该类不含神经网络模型
    """

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, data, pos, length):
        """删除指定位置的单词

        Args:
            data: 形状为[batch_size, max_seq_len]的Tensor，表示语句
            pos: 形状为[batch_size]的Tensor，表示每个语句要删除的位置
            length: 形状为[batch_size]的Tensor，表示每个语句的有效长度

        Returns:
            sentences: 形状为[batch_size, max_seq_len]的Tensor，表示删除指定单词后的语句
            length_updated: 形状为[batch_size]的Tensor，每个位置等于之前的值减一
        """
        assert data.shape[0] > 0
        assert data.shape[0] == pos.shape[0] and pos.shape[0] == length.shape[0]

        batch_size = data.shape[0]
        sentences = torch.zeros_like(data).copy_(data)
        length_updated = torch.zeros_like(length).copy_(length)

        for i in range(batch_size):
            position = pos[i] if pos[i] > -1 else 0
            sentences[i, position: -1] = data[i, position + 1:]
            sentences[i, -1] = self.pad_id
            length_updated[i] = length[i] - 1

        return sentences, length_updated


class OperationBase(nn.Module):
    """
    插入、替换操作的基类
    含有神经网络模型，需要决定插入/替换什么单词
    """

    def __init__(self,
                 hidden_dim,
                 num_layers,
                 dropout_rate,
                 bidirectional,
                 random_sample=False):
        """初始化类

        Args:
            hidden_dim: 循环神经网络的隐层维度
            num_layers: 循环神经网络的层数
            dropout_rate: 循环神经网络的dropout rate
            bidirectional: 是否使用双向神经网络
            random_sample: 决策时是否使用随机采样，仅在训练模式生效
        """
        super(OperationBase, self).__init__()

        self.vocabulary = Vocabulary(config.vocab_file)

        self.vocab_size = self.vocabulary.vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = config.embedding_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.random_sample = random_sample

        self.embedding = nn.Embedding.from_pretrained(self.vocabulary.embedding)
        self.rnn_gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        self.dense = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), self.vocab_size)

    def forward(self, sentences, position, length):
        """前向传播过程，仅计算需要插入/替换的单词

        Args:
            sentences: 需要修改的语句，形状为[batch_size, seq_len]的Tensor
            position: 需要修改的位置，形状为[batch_size]的Tensor
            length: 各语句的有效长度，本函数中没有用到，但在子类的方法中需要

        Returns:
            output_dense: 全连接层（不含非线性激活函数）的输出，是形状为[batch_size, vocab_size]的Tensor
            decision_prob: 经过Softmax的全连接层输出，代表选择各个单词的概率，是形状为[batch_size, vocab_size]的Tensor
            decision_index: 概率最大的单词的编号，即decision_prob取argmax的结果，是形状为[batch_size]的Tensor
        """
        assert sentences.shape[0] > 0

        batch_size = sentences.shape[0]
        init_hidden = self.init_states(batch_size)

        embedded = self.embedding(sentences)  # shape: [batch_size, seq_len, embedding_dim]
        # output shape: [batch_size, seq_len, hidden_dim]
        output, hidden = self.rnn_gru(embedded, init_hidden)

        output_position = []
        for i in range(batch_size):
            output_position.append(output[i, position[i], :])
        output_position = torch.stack(output_position, dim=0)

        output_dense = self.dense(output_position)
        decision_prob = functional.softmax(output_dense, dim=1)

        if self.random_sample and self.training:
            decision_index, decision_probs = sample_2d(decision_prob, temperature=1)
        else:
            decision_probs, decision_index = torch.max(decision_prob, dim=1)

        return output_dense, decision_probs, decision_index

    def init_states(self, batch_size):
        """生成循环神经网络所需的初始状态
        初始状态全取零

        Args:
            batch_size: 数据数量

        Returns:
            state: 合适维度的全零状态
        """
        state = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_dim))
        if config.gpu:
            state = state.cuda()
        return state


class InsertBehind(OperationBase):
    """
    在指定位置后插入的操作
    """

    def __init__(self, hidden_dim, num_layers, dropout_rate, bidirectional, random_sample=False):
        super(InsertBehind, self).__init__(hidden_dim, num_layers, dropout_rate, bidirectional, random_sample)

    def forward(self, sentences, position, length):
        """在取得要插入的单词编号后，在指定位置后插入该单词

        Args:
            sentences: 需要修改的语句，形状为[batch_size, seq_len]的Tensor
            position: 需要修改的位置，形状为[batch_size]的Tensor
            length: 各语句的有效长度

        Returns:
            decision_logits: 全连接层（不含非线性激活函数）的输出，是形状为[batch_size, vocab_size]的Tensor
            decision_prob: 经过Softmax的全连接层输出，代表选择各个单词的概率，是形状为[batch_size, vocab_size]的Tensor
            sentences_inserted: 经过插入单词操作后的语句，是形状为[batch_size, seq_len]的Tensor
            length_inserted: 经过插入单词操作后的各语句有效长度，是形状为[batch_size]的Tensor
        """
        logits, decision_probs, decision_index = super(InsertBehind, self).forward(sentences, position, length)

        sentences_inserted = torch.zeros_like(sentences).copy_(sentences)
        length_inserted = torch.zeros_like(length).copy_(length)
        batch_size = sentences.shape[0]
        for i in range(batch_size):
            pos = position[i]
            if pos == config.max_sentence_length - 2:
                sentences_inserted[i, pos + 1] = decision_index[i]
            elif pos < config.max_sentence_length - 2:
                sentences_inserted[i, pos + 2:] = sentences[i, pos + 1: -1]
                sentences_inserted[i, pos + 1] = decision_index[i]
                if length_inserted[i] < config.max_sentence_length:
                    length_inserted[i] += 1

        return logits, decision_probs, sentences_inserted, length_inserted


class InsertFront(OperationBase):
    """
    在指定位置前插入的操作
    """

    def __init__(self, hidden_dim, num_layers, dropout_rate, bidirectional, random_sample=False):
        super(InsertFront, self).__init__(hidden_dim, num_layers, dropout_rate, bidirectional, random_sample)

    def forward(self, sentences, position, length):
        """在取得要插入的单词编号后，在指定位置前插入该单词

        Args:
            sentences: 需要修改的语句，形状为[batch_size, seq_len]的Tensor
            position: 需要修改的位置，形状为[batch_size]的Tensor
            length: 各语句的有效长度

        Returns:
            decision_logits: 全连接层（不含非线性激活函数）的输出，是形状为[batch_size, vocab_size]的Tensor
            decision_prob: 经过Softmax的全连接层输出，代表选择各个单词的概率，是形状为[batch_size, vocab_size]的Tensor
            sentences_inserted: 经过插入单词操作后的语句，是形状为[batch_size, seq_len]的Tensor
            length_inserted: 经过插入单词操作后的各语句有效长度，是形状为[batch_size]的Tensor
        """
        logits, decision_probs, decision_index = super(InsertFront, self).forward(sentences, position, length)

        sentences_inserted = torch.zeros_like(sentences).copy_(sentences)
        length_inserted = torch.zeros_like(length).copy_(length)
        batch_size = sentences.shape[0]
        for i in range(batch_size):
            pos = position[i]
            sentences_inserted[i, pos + 1:] = sentences[i, pos: -1]
            sentences_inserted[i, pos] = decision_index[i]
            if length_inserted[i] < config.max_sentence_length:
                length_inserted[i] += 1

        return logits, decision_probs, sentences_inserted, length_inserted


class Replace(OperationBase):
    """
    在指定位置进行替换的操作
    """

    def __init__(self, hidden_dim, num_layers, dropout_rate, bidirectional, random_sample=False):
        super(Replace, self).__init__(hidden_dim, num_layers, dropout_rate, bidirectional, random_sample)

    def forward(self, sentences, position, length):
        """在取得要插入的单词编号后，在指定位置替换该单词

        Args:
            sentences: 需要修改的语句，形状为[batch_size, seq_len]的Tensor
            position: 需要修改的位置，形状为[batch_size]的Tensor
            length: 各语句的有效长度

        Returns:
            decision_logits: 全连接层（不含非线性激活函数）的输出，是形状为[batch_size, vocab_size]的Tensor
            decision_prob: 经过Softmax的全连接层输出，代表选择各个单词的概率，是形状为[batch_size, vocab_size]的Tensor
            sentences_inserted: 经过替换单词操作后的语句，是形状为[batch_size, seq_len]的Tensor
            length: 经过替换单词操作后的各语句有效长度，和初始长度相同
        """
        logits, decision_probs, decision_index = super(Replace, self).forward(sentences, position, length)

        sentences_replaced = torch.zeros_like(sentences).copy_(sentences)
        batch_size = sentences.shape[0]
        for i in range(batch_size):
            pos = position[i]
            sentences_replaced[i, pos] = decision_index[i]

        return logits, decision_probs, sentences_replaced, length


if __name__ == '__main__':
    pass
