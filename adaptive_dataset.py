# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
from config import Config
from dataset import DualNovelDataSet, Vocabulary


config = Config()


class AdaptiveDataSet(data.Dataset):
    """
    可扩充数据集
    可以利用训练过程中产生的句子来增加数据集中的数据量
    """

    def __init__(self, max_iter=2):
        """初始化类
        从初始数据集中读取数据来形成最初的数据

        Args:
            max_iter: 最多允许修改多少次的数据重新加入数据集
        """
        super(AdaptiveDataSet, self).__init__()

        self.vocabulary = Vocabulary(config.vocab_file)
        self.primitive_dataset = DualNovelDataSet()

        self.max_iter = max_iter
        self.current_iter = 0
        self.data_0 = []
        self.data_1 = []

        # 设置特殊符号
        self.pad = self.vocabulary.word2id['<pad>']
        self.go = self.vocabulary.word2id['<go>']
        self.eos = self.vocabulary.word2id['<eos>']
        self.unk = self.vocabulary.word2id['<unk>']

        # 读取初始数据
        self.initialize()

    def initialize(self):
        """
        从原始数据集中读取数据
        """
        for style, styled_dataset in enumerate(self.primitive_dataset.data_index):
            for sentence in styled_dataset:
                sentence_tensor, _, _, length_tensor = self.primitive_dataset.process_sentence_classification(sentence)
                sample = {
                    'sentence': sentence_tensor,
                    'length': length_tensor,
                    'style': style,
                    'iter': 0
                }

                if style == 0:
                    self.data_0.append(sample)
                elif style == 1:
                    self.data_1.append(sample)

    def add_data(self, sentences, length, style, iter_prev):
        """向数据集中增加一批样本

        Args:
            sentences: 形状为[batch_size, max_seq_len]的LongTensor，表示要增加的语句
            length: 形状为[batch_size]的LongTensor，表示语句对应的有效长度
            style: 形状为[batch_size]的Tensor，表示语句对应的风格
            iter_prev: 形状为[batch_size]的Tensor，表示修改前的语句修改次数
        """
        if self.current_iter >= self.max_iter:
            return

        batch_size = sentences.shape[0]
        assert length.shape[0] == batch_size and style.shape[0] == batch_size and iter_prev.shape[0] == batch_size

        for i in range(batch_size):
            if iter_prev[i] < self.max_iter:
                sample = {
                    'sentence': sentences[i],
                    'length': length[i],
                    'style': style[i],
                    'iter': iter_prev[i].item() + 1
                }

                if style[i].item() == 0:
                    self.data_0.append(sample)
                elif style[i].item() == 1:
                    self.data_1.append(sample)
                else:
                    raise ValueError('Unsupported style: {}'.format(style[i]))

    def __getitem__(self, index):
        """取数据集中指定编号的数据

        Args:
            index: 数据的编号

        Returns:
            sentence: 形状为[max_seq_len]的LongTensor，表示语句
            length: 形状为[1]的Tensor，表示语句的有效长度
            style: 形状为[1]的Tensor，表示语句的风格，值取0或1
            iter: 形状为[1]的Tensor，表示该语句已经修改了多少次
        """
        assert index < len(self.data_0) and index < len(self.data_1)

        sample_0 = self.data_0[index]
        sample_1 = self.data_1[index]

        return (sample_0['sentence'], sample_0['length'], sample_0['iter']), \
               (sample_1['sentence'], sample_1['length'], sample_1['iter'])

    def __len__(self):
        """获取数据集中数据的数量

        Returns:
            len: 数据样本的数量
        """
        return min(len(self.data_0), len(self.data_1))

    def get_sentence(self, data, length=None):
        """根据表示语句的LongTensor得到对应的文字
        如果length为None则自动截掉<pad>的部分

        Args:
            data: 表示语句的Tensor，形状为[batch_size, max_seq_len]
            length: (可选)语句的有效长度，形状为[batch_size]

        Returns:
            sentences: 得到的文字语句，是一个长度为batch_size的列表，每个元素是由单词组成的列表
        """
        batch_size = data.shape[0]

        if length is not None:
            assert length.shape[0] == batch_size

        sentences = []
        for i in range(batch_size):
            sentence = []
            if length is not None:
                for j in range(length[i].item()):
                    sentence.append(self.vocabulary.id2word[data[i, j].item()])
            else:
                for j in range(config.max_sentence_length):
                    word = self.vocabulary.id2word[data[i, j].item()]
                    if word != self.pad:
                        sentence.append(word)
                    else:
                        break
            sentences.append(sentence)

        return sentences
