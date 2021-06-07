# -*- coding: utf-8 -*-
"""
实现了词汇表类，其中存储了字典word2id, id2word以及embedding的信息
如果有现成的embedding结果则从文件中读取，没有则随机生成
"""
import pickle
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from config import Config


config = Config()


def sample_2d(probability, temperature):
    """根据概率进行随机采样，可以用于替代确定性的argmax操作

    Args:
        probability: 形状为[batch_size, n]的Tensor
        temperature: 取1时，将使用temperature机制来增大随机性

    Returns:
        sample_index: 形状为[batch_size]的Tensor，表示每个位置采样的样本编号
        sample_probs: 形状为[batch_size]的Tensor，表示选取的位置对应的概率
    """
    if temperature != 1:
        temp = torch.exp(torch.div(torch.log(probability + 1e-10), config.temp_att))
    else:
        temp = probability

    sample_index = torch.multinomial(temp, 1)  # shape: [batch_size, 1]
    sample_probs = probability.gather(1, sample_index)  # shape: [batch_size, 1]
    sample_index = sample_index.squeeze(1)  # shape: [batch_size]
    sample_probs = sample_probs.squeeze(1)  # shape: [batch_size]

    return sample_index, sample_probs


class Vocabulary(object):
    def __init__(self, vocab_file):
        with open(vocab_file, 'rb') as f:
            self.vocab_size, self.word2id, self.id2word = pickle.load(f)
        self.embedding = np.random.random_sample((self.vocab_size, config.embedding_dim))

        if config.embedding_file:
            self.embedding = np.load(config.embedding_file)
            assert self.embedding.shape[0] == self.vocab_size and self.embedding.shape[1] == config.embedding_dim

        for i in range(self.vocab_size):
            self.embedding[i] /= np.linalg.norm(self.embedding[i])

        self.embedding = torch.FloatTensor(self.embedding)


class DualNovelDataSet(data.Dataset):
    """
    同时含有正负样本的数据集
    主要为适应DataLoader类而建立
    输出是带标签的
    实际测试时应使用SingleNovelDataSet
    """

    def __init__(self,
                 make_up=True,
                 test=False,
                 max_len=30,
                 classification=True,
                 verbose=False):
        """初始化类，处理文本数据，并根据vocabulary将单词转换为对应的编号

        Args:
            make_up: 是否补充数据集长度，使含有正负样本的数据集长度相同
            test: 是否是测试模式（目前该参数无影响）
            max_len: 语句的最大长度，如果长度不足将补<pad>符号，长度过大则将截断
            classification: 是否是分类器模式，如果为False则生成的数据将用于language model
            verbose: 初始化时是否输出提示信息
        """
        super(DualNovelDataSet, self).__init__()

        self.test = test
        self.classification = classification

        # 从文件中读取处理过的文本
        self.data_path = [config.style_0_data_path, config.style_1_data_path]
        self.data_text = []
        for path in self.data_path:
            tmp_data = []
            with open(path, 'r') as f:
                for line in f:
                    sentence = line.split()
                    tmp_data.append(sentence)
            self.data_text.append(tmp_data)

        # 如果make_up=True，则通过重复添加的方式让两个数据集长度相同
        if make_up:
            if len(self.data_text[0]) < len(self.data_text[1]):
                self.data_text[0] = self.make_up_data(self.data_text[0], len(self.data_text[1]))
            else:
                self.data_text[1] = self.make_up_data(self.data_text[1], len(self.data_text[0]))
            assert len(self.data_text[0]) == len(self.data_text[1])

        # 根据vocabulary类，将单词转换为对应的编号
        self.vocabulary = Vocabulary(config.vocab_file)
        self.max_len = max_len
        self.data_index = []
        for data_i in self.data_text:
            self.data_index.append([
                [
                    self.vocabulary.word2id[word]
                    if word in self.vocabulary.word2id
                    else self.vocabulary.word2id['<unk>']
                    for word in sentence
                ]
                for sentence in data_i
            ])

        # 设置特殊符号
        self.pad = self.vocabulary.word2id['<pad>']
        self.go = self.vocabulary.word2id['<go>']
        self.eos = self.vocabulary.word2id['<eos>']
        self.unk = self.vocabulary.word2id['<unk>']

        # 输出提示信息
        if verbose:
            print('***** Novel Dataset established *****')
            print('Vocabulary size: {}'.format(self.vocabulary.vocab_size))
            print('Sentences in each dataset: {}, {}'.format(len(self.data_text[0]), len(self.data_text[1])))
            print('Maximum sentence length: {}'.format(self.max_len))

    def make_up_data(self, data_list, target_len):
        """通过重复添加的方式补充数据集的长度

        Args:
            data_list: 含有数据的列表
            target_len: 目标长度，应大于列表长度
        """
        assert target_len >= len(data_list)
        return [data_list[i % len(data_list)] for i in range(target_len)]

    def process_sentence_classification(self, sentence):
        """处理单个语句，得到不同形式的数据，特性如下
        bare: 仅将原语句添加<pad>至最大长度
        go: 将原语句添加<pad>至最大长度，并在前面添加其实符号<go>
        eos: 将语句末尾添加结束符号<eos>，并添加<pad>至最大长度

        Args:
            sentence: 表示语句，一个list，各元素是各单词的编号

        Returns:
            bare_sentence: bare格式的语句转换为的Tensor，格式同输入
            go_sentence: go格式的语句对应的Tensor
            eos_sentence: eos格式的语句对应的Tensor
            length: 仅含单个元素的Tensor，代表语句的有效长度
        """
        length = len(sentence)
        if length >= self.max_len:
            sentence = sentence[:self.max_len]
            length = self.max_len
            padding = []
        elif length < self.max_len:
            padding = [self.pad] * (self.max_len - length)

        bare_sentence = torch.LongTensor(sentence + padding)
        go_sentence = torch.LongTensor([self.go] + sentence + padding)
        eos_sentence = torch.LongTensor(sentence + [self.eos] + padding)
        return bare_sentence, go_sentence, eos_sentence, torch.LongTensor([length]).squeeze()

    def process_sentence_lm(self, sentence):
        length = len(sentence)
        if length >= self.max_len + 1:
            sentence = sentence[:self.max_len + 1]
            length = self.max_len
            padding = []
        elif length <= self.max_len + 1:
            if length == self.max_len + 1:
                length = self.max_len
            padding = [self.pad] * (self.max_len + 1 - length)

        padded_sentence = sentence + padding  # length: self.max_len + 1
        assert len(padded_sentence) == self.max_len + 1

        inputs = torch.LongTensor(padded_sentence[:self.max_len])
        targets = torch.LongTensor(padded_sentence[1:])

        return inputs, targets, torch.LongTensor([length]).squeeze()

    def get_sentences(self, data):
        if len(data.shape) == 1:
            return [self.vocabulary.id2word[data[i]] for i in range(data.shape[0])]
        elif len(data.shape) == 2:
            return [[self.vocabulary.id2word[data[i, j]] for j in range(data.shape[1])] for i in range(data.shape[0])]
        else:
            raise ValueError('Dimension of data must be 1 or 2')

    def __getitem__(self, index):
        """返回指定编号的一组正、负样本
        每一个样本都按照self.process_sentence函数的返回值格式来返回

        Args:
            index: 样本的编号

        Returns:
            self.classification == True: 分类器模式
                (bare_0, go_0, eos_0, len_0): 负样本（风格为0）的数据
                (bare_1, go_1, eos_1, len_1): 正样本（风格为1）的数据
            self.classification == False: 语言模型模式
                (input_0, label_0, len_0): 负样本（风格为0）的数据
                (input_1, label_1, len_1): 正样本（风格为1）的数据
        """
        assert index < len(self.data_index[0])

        sentence_0 = self.data_index[0][index]
        sentence_1 = self.data_index[1][index]

        if self.classification:
            bare_0, go_0, eos_0, len_0 = self.process_sentence_classification(sentence_0)
            bare_1, go_1, eos_1, len_1 = self.process_sentence_classification(sentence_1)
            return (bare_0, go_0, eos_0, len_0), (bare_1, go_1, eos_1, len_1)
        else:
            input_0, target_0, len_0 = self.process_sentence_lm(sentence_0)
            input_1, target_1, len_1 = self.process_sentence_lm(sentence_1)
            return (input_0, target_0, len_0), (input_1, target_1, len_1)

    def get_text_item(self, index):
        sentence_0 = self.data_text[0][index]
        sentence_1 = self.data_text[1][index]
        return sentence_0, sentence_1

    def __len__(self):
        """返回数据集中正/负样本的数量
        因为同时含有正负样本，数据集中数据样本的实际数量应是返回值的两倍

        Returns:
            length: 数据集中正/负样本的数量（正负样本是一样多的）
        """
        return len(self.data_index[0])


class SingleNovelDataSet(object):
    def __init__(self, data_path, name, make_up_len=-1, test=False, max_len=30):

        super(SingleNovelDataSet, self).__init__()

        self.test = test
        self.name = name

        # 从文件中读取处理过的文本
        self.data_text = []
        with open(data_path, 'r') as f:
            for line in f:
                sentence = line.split()
                self.data_text.append(sentence)

        # 如果make_up=True，则通过重复添加的方式让两个数据集长度相同
        if make_up_len > len(self.data_text):
            self.data_text = [self.data_text[i % len(self.data_text)] for i in range(make_up_len)]

        # 根据vocabulary类，将单词转换为对应的编号
        self.vocabulary = Vocabulary(config.vocab_file)
        self.max_len = max_len
        self.data_index = [
            [
                self.vocabulary.word2id[word]
                if word in self.vocabulary.word2id
                else self.vocabulary.word2id['<unk>']
                for word in sentence
            ]
            for sentence in self.data_text
        ]

        # 设置特殊符号
        self.pad = self.vocabulary.word2id['<pad>']
        self.go = self.vocabulary.word2id['<go>']
        self.eos = self.vocabulary.word2id['<eos>']
        self.unk = self.vocabulary.word2id['<unk>']

        # 输出提示信息
        print('***** Novel Dataset {} established *****'.format(self.name))
        print('Vocabulary size: {}'.format(self.vocabulary.vocab_size))
        print('Sentences in each dataset: {}, {}'.format(len(self.data_text[0]), len(self.data_text[1])))
        print('Maximum sentence length: {}'.format(self.max_len))

    def process_sentence(self, sentence):
        length = len(sentence)
        if length >= self.max_len:
            sentence = sentence[:self.max_len]
            padding = []
            length = self.max_len
        elif length < self.max_len:
            padding = [self.pad] * (self.max_len - length)

        bare_sentence = torch.LongTensor(sentence + padding)
        go_sentence = torch.LongTensor([self.go] + sentence + padding)
        eos_sentence = torch.LongTensor(sentence + [self.eos] + padding)
        return bare_sentence, go_sentence, eos_sentence, torch.LongTensor([length]).squeeze()

    def __getitem__(self, index):
        sentence = self.data_index[index]
        bare, go, eos, length = self.process_sentence(sentence)

        return bare, go, eos, length

    def __len__(self):
        return len(self.data_index)


if __name__ == '__main__':
    dataset = DualNovelDataSet(classification=False)

    loader = DataLoader(dataset, batch_size=16)
    for data in loader:
        (bare_0, label_0, len_0), (bare_1, label_1, len_1) = data
        print(bare_0)
        print(label_0)
        print(bare_0.shape)
        print(len_0)
        print(len_0.shape)
        break
