# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import Config
from dataset import Vocabulary, DualNovelDataSet


config = Config()


class PointerModule(object):
    """
    实现的带Attention机制的风格分类器
    本类结构与AuxiliaryStyleClassifier相同，只是使用的网络结构不同
    """

    def __init__(self):
        """
        初始化Attention分类器
        主要步骤包括读取数据、建立模型、配置数据集和优化器、设定参数
        因为模型结构不同，类不方便直接复用，因此再次实现了一遍
        """
        # 读取词汇和数据
        self.vocabulary = Vocabulary(config.vocab_file)
        self.att_embedding = nn.Embedding.from_pretrained(self.vocabulary.embedding, freeze=False)
        if config.gpu:
            self.att_embedding = self.att_embedding.cuda()

        # 建立AttentionClassifier模型
        self.attention_classifier = AttentionClassifier(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
            bidirectional=config.bidirectional
        )
        if config.gpu:
            self.attention_classifier = self.attention_classifier.cuda()

        # 设定训练和测试数据集
        self.train_set = DualNovelDataSet(test=False, max_len=config.max_sentence_length)
        self.test_set = DualNovelDataSet(test=True, max_len=config.max_sentence_length)

        # 设定可训练变量和优化器
        self.trainable_variables = []
        for k, v in self.att_embedding.state_dict(keep_vars=True).items():
            if v.requires_grad:
                self.trainable_variables.append(v)
        for k, v in self.attention_classifier.state_dict(keep_vars=True).items():
            if v.requires_grad:
                self.trainable_variables.append(v)

        self.learning_rate = config.pointer_learning_rate
        self.beta1 = config.pointer_beta1
        self.beta2 = config.pointer_beta2

        self.optimizer = Adam(self.trainable_variables, self.learning_rate, (self.beta1, self.beta2))

        # 设定训练参数
        self.batch_size = config.pointer_batch_size
        self.epochs = config.pointer_epochs
        self.num_workders = config.pointer_num_workers

    def set_training(self, train_mode):
        """设定训练/测试模式

        Args:
            train_mode: 布尔型，是否是训练模式
        """
        self.att_embedding.train(train_mode)
        self.attention_classifier.train(train_mode)

    def save_model(self,
                   embedding_path=config.pointer_embedding_model,
                   att_classifier_path=config.pointer_att_classifier_model):
        """将模型保存到指定路径

        Args:
            embedding_path: 保存Embedding层参数的路径
            att_classifier_path: 保存分类器参数的路径
        """
        torch.save(self.attention_classifier.state_dict(), att_classifier_path)
        torch.save(self.att_embedding.state_dict(), embedding_path)

    def load_model(self,
                   embedding_path=config.pointer_embedding_model,
                   att_classifier_path=config.pointer_att_classifier_model):
        """从指定路径的文件读取模型参数

        Args:
            embedding_path: 保存Embedding层参数的路径
            att_classifier_path: 保存分类器参数的路径
        """
        self.attention_classifier.load_state_dict(
            torch.load(att_classifier_path, map_location=lambda storage, loc: storage)
        )
        self.att_embedding.load_state_dict(
            torch.load(embedding_path, map_location=lambda storage, loc: storage)
        )

    def train(self, verbose=False, graph=False):
        """训练辅助分类器

        Args:
            verbose: 是否输出提示信息，即每个Epoch结束后输出该代的损失
            graph: 训练结束后是否显示损失变化曲线图
        """
        loss_list = []
        acc_list = []
        for epoch in range(self.epochs):
            epoch_loss = self.run_epoch(test=False)
            loss_list += epoch_loss
            if verbose:
                print('\n[TRAIN] Epoch {}, mean loss {}'.format(epoch, np.mean(epoch_loss)))

            train_accuracy = self.run_epoch(test=True)
            acc_list.append(train_accuracy)

        if graph:
            plt.figure()

            plt.subplot(1, 2, 1)
            plt.plot([x for x in range(len(loss_list))], loss_list)
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.grid()
            plt.title('Training loss')

            plt.subplot(1, 2, 2)
            plt.plot([x for x in range(len(acc_list))], acc_list)
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.grid()
            plt.title('Training accuracy')

            plt.show()

    def visualize_attention(self, index, test=False):
        """将Attention机制的效果进行可视化

        Args:
            index: 数据样本的编号
            test: 是否使用测试集，布尔型
        """
        if test:
            (bare_0, _, _, len_0), (bare_1, _, _, len_1) = self.test_set.__getitem__(index)
            text_0, text_1 = self.test_set.get_text_item(index)
        else:
            (bare_0, _, _, len_0), (bare_1, _, _, len_1) = self.train_set.__getitem__(index)
            text_0, text_1 = self.train_set.get_text_item(index)

        #print(bare_0)
        #print(type(bare_0))
        sentences = torch.cat([bare_0.unsqueeze(0), bare_1.unsqueeze(0)], dim=0)
        length = torch.cat([len_0.unsqueeze(0), len_1.unsqueeze(0)], dim=0)

        if config.gpu:
            sentences = sentences.cuda()
            length = length.cuda()

        null_mask = sentences.eq(self.train_set.pad)
        print(sentences)
        #print(length)
        #print(null_mask)

        print(text_0, text_1)

        embedded = self.att_embedding(sentences)
        classes, attention_weights = self.attention_classifier(embedded, length, null_mask)
        classes = functional.sigmoid(classes)
        print(classes)
        print(attention_weights)

    def inference(self, sentences, length, null_mask=None):
        """推断步骤，根据输入的语句判断类别并给出attention权重

        Args:
            sentences: 形状为[batch_size, max_seq_len]的LongTensor，表示输入的语句
            length: 形状为[batch_size]的LongTensor，表示各语句的有效长度
            null_mask: 形状为[batch_size, max_seq_len]的Tensor，表示pad符号的蒙版，如果不给出则将根据sentences计算

        Returns:
            class_probs: 形状为[batch_size]的Tensor，表示各语句的类别（0-1之间）
            class_logits: 形状为[batch_size]的Tensor，是class_probs经过sigmoid之前的结果
            attention_weights: Attention层的权重
        """
        if null_mask is None:
            null_mask = sentences.eq(self.train_set.pad)

        with torch.no_grad():
            embedded = self.att_embedding(sentences)
            class_logits, attention_weights = self.attention_classifier(embedded, length, null_mask)
            class_probs = functional.sigmoid(class_logits)

        return class_probs, class_logits, attention_weights

    def run_epoch(self, test=False):
        """运行一个epoch，可以指定是训练还是测试模式

        Args:
            test: 是否是测试模式
        """
        loss_list = []
        acc_list = []

        if not test:
            loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workders)
            self.attention_classifier.train(mode=True)
            self.att_embedding.train(mode=True)
        else:
            loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workders)
            self.attention_classifier.train(mode=False)
            self.att_embedding.train(mode=False)

        with tqdm(loader) as pbar:
            for data in pbar:
                sentences, labels, length = self.preprocess_data(data)
                null_mask = sentences.eq(self.train_set.pad)

                embedded = self.att_embedding(sentences)
                classes, attention_weights = self.attention_classifier(embedded, length, null_mask)
                loss = functional.binary_cross_entropy_with_logits(classes, labels)

                if not test:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_list.append(loss.item())
                else:
                    classes = functional.sigmoid(classes)
                    prediction = (classes > 0.5).float()

                    num_items = sentences.shape[0]
                    num_wrong = torch.abs(prediction - labels).sum().item()
                    accuracy = 1.0 - float(num_wrong) / float(num_items)

                    acc_list.append(accuracy)

        return np.mean(acc_list) if test else loss_list

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
        length = torch.cat([len_0, len_1], dim=0)

        if config.gpu:
            sentences = sentences.cuda()
            label = label.cuda()

        return sentences, label, length


class AttentionClassifier(nn.Module):
    """
    实现带有attention机制的分类器，在分类的同时判断单词的重要性，训练过程中作pointer使用
    最后一层是没有非线性激活函数的，如果需要取输出需要手动加sigmoid
    """

    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout_rate, bidirectional):
        """初始化类

        Args:
            embedding_dim: embedding层的维度
            hidden_dim: 循环神经网络中隐层的维度
            num_layers: 循环神经网络的层数
            dropout_rate: dropout层的几率
            bidirectional: 布尔型，表示是否使用双向神经网络
        """
        super(AttentionClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.gru_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=bidirectional
        )
        self.attention_module = AttentionUnit(
            query_dim=hidden_dim * self.directions,
            key_dim=hidden_dim * self.directions,
            attention_dim=self.hidden_dim
        )
        # self.dense = nn.Sequential(
        #     nn.Linear(self.hidden_dim * self.directions, 1),
        #     nn.Sigmoid()
        # )
        self.dense = nn.Linear(self.hidden_dim * self.directions, 1)

    def forward(self, inputs, length, null_mask):
        """前向运算

        Args:
            inputs: 输入的语句信息, 形状为[batch_size, max_seq_len, embedding_dim]
            length: 用于pack的参数
            null_mask: 形状为[batch_size, max_seq_len], 指示无效位置的蒙版

        Returns:
            classes: 分类结果, 形状为[batch_size]
            attention_weights: attention权重, 形状为[batch_size, max_seq_len], 受null_mask影响
        """
        batch_size = inputs.shape[0]
        max_seq_len = inputs.shape[1]

        inputs = inputs.transpose(0, 1)  # shape: [max_seq_len, batch_size, embedding_dim]
        packed_input = pack(inputs, length, enforce_sorted=False)
        # hidden shape: [num_layers * directions, batch_size, hidden_dim]
        outputs, hidden = self.gru_encoder(packed_input)
        # shape: [max_seq_len, batch_size, hidden_dim * directions]
        outputs = unpack(outputs, total_length=max_seq_len)[0]
        # hidden shape: [self.num_layers, batch_size, hidden_dim * directions]
        hidden = hidden.view(
            self.num_layers, self.directions, batch_size, self.hidden_dim
        ).transpose(1, 2).transpose(2, 3).contiguous().view(self.num_layers, batch_size, -1)
        hidden = hidden[-1, :, :]  # shape: [batch_size, hidden_dim * directions]

        # context shape: [batch_size, hidden_dim * directions]
        # attention_weights shape: [batch_size, max_seq_len]
        context, attention_weights = self.attention_module(
            queries=hidden,
            keys=outputs.transpose(0, 1),
            null_mask=null_mask
        )

        classes = self.dense(context).squeeze(1)  # shape: [batch_size]

        return classes, attention_weights


class AttentionUnit(nn.Module):
    """
    实现Attention机制的模块
    """

    def __init__(self, query_dim, key_dim, attention_dim):
        super(AttentionUnit, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.attention_dim = attention_dim

        self.query_dense = nn.Linear(query_dim, attention_dim)
        self.keys_dense = nn.Linear(key_dim, attention_dim)
        self.alpha_dense = nn.Linear(attention_dim, 1)

    def forward(self, queries, keys, null_mask):
        """Attention模块的前向传播

        Args:
            queries: 形状为[batch_size, query_dim]的Tensor
            keys: 形状为[batch_size, num_keys, key_dim]的Tensor
            null_mask: 形状为[batch_size, num_keys]的Tensor, 指定位置经过softmax会变为0, 效果为丢弃指定位置keys的作用

        Returns:
            attended_keys: 经过Attention作用的keys
            attention_weights: Attention的权重
        """
        assert queries.shape[0] == keys.shape[0]
        num_keys = keys.shape[1]

        t_query = self.query_dense(queries)  # shape: [-1, attention_dim]
        t_query = t_query.unsqueeze(1).expand(-1, num_keys, -1)  # shape: [-1, num_keys, attention_dim]
        t_key = self.keys_dense(keys)  # shape: [-1, num_keys, attention_dim]

        alpha = self.alpha_dense(torch.tanh(t_query + t_key)).squeeze(2)  # shape: [-1, num_keys]
        alpha.masked_fill_(null_mask, -float('inf'))  # null_mask覆盖的地方赋为负无穷

        attention_weights = functional.softmax(alpha, dim=1)
        attended_keys = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)  # shape: [-1, key_dim]

        return attended_keys, attention_weights


def pre_train_pointer():
    print('Training pointer model...')
    model = PointerModule()
    model.train(verbose=True, graph=True)
    model.save_model(config.pointer_embedding_model, config.pointer_att_classifier_model)
    print('Pointer model saved.')

    # model.load_model(config.pointer_embedding_model, config.pointer_att_classifier_model)
    # model.visualize_attention(0, test=False)


if __name__ == '__main__':
    pre_train_pointer()
