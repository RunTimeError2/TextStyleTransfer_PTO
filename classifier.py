# -*- coding: utf-8 -*-
"""
实现了基于TextCNN的辅助分类器，将在训练过程中用到
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as functional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataset import Vocabulary, DualNovelDataSet, SingleNovelDataSet
from config import Config


config = Config()


class AuxiliaryStyleClassifier(object):
    def __init__(self):
        """
        初始化辅助分类器
        主要步骤包括读取文本和词汇数据，建立分类器模型，配置数据集和优化器，设定训练所需参数
        """
        # 读取词汇和Embedding
        self.vocabulary = Vocabulary(config.vocab_file)
        self.embedding = nn.Embedding.from_pretrained(self.vocabulary.embedding, freeze=False)
        if config.gpu:
            self.embedding = self.embedding.cuda()

        # 建立分类器
        self.classifier = TextCNNClassifier(
            kernels=config.text_cnn_kernels,
            conv_channels=config.text_cnn_channels,
            embedding_dim=config.embedding_dim,
            dropout_rate=config.text_cnn_dropout_rate
        )
        if config.gpu:
            self.classifier = self.classifier.cuda()

        # 读取数据集
        self.train_set = DualNovelDataSet(test=False, max_len=config.max_sentence_length)
        self.test_set = DualNovelDataSet(test=True, max_len=config.max_sentence_length)

        # 配置优化器
        self.trainable_variables = []
        for k, v in self.classifier.state_dict(keep_vars=True).items():
            if v.requires_grad:
                self.trainable_variables.append(v)
        for k, v in self.embedding.state_dict(keep_vars=True).items():
            if v.requires_grad:
                self.trainable_variables.append(v)

        self.learning_rate = config.text_cnn_learning_rate
        self.beta1 = config.text_cnn_beta1
        self.beta2 = config.text_cnn_beta2

        self.optimizer = Adam(self.trainable_variables, self.learning_rate, (self.beta1, self.beta2))

        # 配置训练所需参数
        self.batch_size = config.text_cnn_batch_size
        self.epochs = config.text_cnn_epochs
        self.num_workers = config.text_cnn_num_workers

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

    def set_training(self, train_mode):
        """设定训练/测试模式

        Args:
            train_mode: 布尔型，是否是训练模式
        """
        self.embedding.train(train_mode)
        self.classifier.train(train_mode)

    def test(self):
        """
        测试辅助分类器
        实际计算的是训练集上的准确率
        """
        mean_accuracy = self.run_epoch(test=True)
        print('\n[TEST] mean accuracy {}'.format(mean_accuracy))

    def save_model(self,
                   embedding_path=config.text_cnn_embedding_model,
                   classifier_path=config.text_cnn_classifier_model):
        """将模型保存到指定路径

        Args:
            embedding_path: 保存Embedding层参数的路径
            classifier_path: 保存分类器参数的路径
        """
        torch.save(self.classifier.state_dict(), classifier_path)
        torch.save(self.embedding.state_dict(), embedding_path)

    def load_model(self,
                   embedding_path=config.text_cnn_embedding_model,
                   classifier_path=config.text_cnn_classifier_model):
        """从指定路径的文件读取模型参数

        Args:
            embedding_path: 保存Embedding层参数的路径
            classifier_path: 保存分类器参数的路径
        """
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=lambda storage, loc: storage))
        self.embedding.load_state_dict(torch.load(embedding_path, map_location=lambda storage, loc: storage))

    def inference(self, sentences):
        """推断步骤，根据输入的语句计算各语句的类别

        Args:
            sentences: 形状为[batch_size, max_seq_len]的LongTensor，表示语句

        Returns:
            class_probs: 形状为[batch_size]的Tensor，表示各语句的类别（0-1之间）
            class_logits: 形状为[batch_size]的Tensor，是class_probs经过sigmoid之前的结果
        """
        with torch.no_grad():
            embedded = self.embedding(sentences)
            class_logits = self.classifier(embedded).squeeze(1)
            class_probs = functional.sigmoid(class_logits)

        return class_probs, class_logits

    def run_epoch(self, test=False):
        """运行一个epoch，可以指定是训练还是测试模式

        Args:
            test: 是否是测试模式
        """
        loss_list = []
        acc_list = []

        if not test:
            loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.classifier.train(mode=True)
            self.embedding.train(mode=True)
        else:
            loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            self.classifier.train(mode=False)
            self.embedding.train(mode=False)

        with tqdm(loader) as pbar:
            for data in pbar:
                sentences, labels = self.preprocess_data(data)

                embedded = self.embedding(sentences)
                classes = self.classifier(embedded).squeeze(1)
                loss = functional.binary_cross_entropy_with_logits(classes, labels)

                if not test:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_list.append(loss.item())
                    pbar.set_description('Current loss {}'.format(round(loss.item(), 4)))
                else:
                    classes = functional.sigmoid(classes)
                    prediction = (classes > 0.5).float()

                    num_items = sentences.shape[0]
                    num_wrong = torch.abs(prediction - labels).sum().item()
                    accuracy = 1.0 - float(num_wrong) / float(num_items)

                    acc_list.append(accuracy)
                    pbar.set_description('Current accuracy: {}'.format(accuracy))

        return np.mean(acc_list) if test else loss_list

    def test_on_dataset(self, data_path):
        """在指定数据集上进行测试，并用图像表示模型的输出结果

        Args:
            data_path: 数据集的路径，其格式是预处理好的文本数据
        """
        data_set = SingleNovelDataSet(
            data_path=data_path, name='test_dataset', test=True, max_len=config.max_sentence_length
        )
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        classes_list = []

        with tqdm(data_loader) as pbar:
            for data in pbar:
                sentences, _, _, _ = data

                embedded = self.embedding(sentences)
                classes = self.classifier(embedded).squeeze(1)
                classes = functional.sigmoid(classes)
                classes_list += classes.tolist()

        plt.figure()
        plt.plot([x for x in range(len(classes_list))], classes_list)
        plt.xlabel('sample id')
        plt.ylabel('style output')
        plt.grid()
        plt.title('Testing results using {}'.format(data_path))
        plt.show()

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


class TextCNNClassifier(nn.Module):
    """
    基于TextCNN的风格分类器
    最后一层是没有非线性激活函数的，如果需要取输出需要手动加sigmoid
    """

    def __init__(self,
                 kernels,
                 conv_channels,
                 embedding_dim,
                 dropout_rate):
        """初始化TextCNN模块

        Args:
            kernels: 卷积核数量
            conv_channels: 卷积的输出通道数
            embedding_dim: embedding层的输出维度，决定输入数据的宽度
            dropout_rate: dropout层的dropout几率
        """
        super(TextCNNClassifier, self).__init__()

        self.kernels = kernels
        self.conv_channels = conv_channels
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.conv_blocks = []
        for kernel in self.kernels:
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=conv_channels,
                    kernel_size=(kernel, self.embedding_dim),
                    padding=(0, 0),
                    stride=(1, 1)
                ),
                nn.LeakyReLU(0.01)
            )
            self.conv_blocks.append(block)

        self.dropout = nn.Dropout(self.dropout_rate)
        # self.dense = nn.Sequential(
        #    nn.Linear(len(self.kernels) * self.conv_channels, 1),
        #     nn.Sigmoid()
        # )
        self.dense = nn.Linear(len(self.kernels) * self.conv_channels, 1)

    def forward(self, x):
        """TextCNN的前向传播

        Args:
            x: 形状为[-1, max_len, embedding_dim]的Tensor

        Returns:
            output: 形状为[-1, 1]的Tensor
        """
        x = x.unsqueeze(1)  # shape: [-1, 1, max_len, embedding_dim]
        conv_outputs = []
        for block in self.conv_blocks:
            conv_output = block(x)  # shape: [-1, conv_channels, max_len, embedding_dim]
            pooled, _ = torch.max(conv_output, dim=2)  # shape: [-1, conv_channels, 1]
            pooled = pooled.squeeze(2)  # shape: [-1, conv_channels]
            conv_outputs.append(pooled)

        output = torch.cat(conv_outputs, dim=1)  # shape: [-1, len(kernels) * conv_channels)
        output = self.dropout(output)
        output = self.dense(output)

        return output


def train_text_cnn():
    """
    训练TextCNN，绘制其训练过程中的损失函数变化图
    并在指定的测试数据集上进行测试，绘制测试结果
    """
    print('Training auxiliary classifier model...')
    model = AuxiliaryStyleClassifier()
    model.train(verbose=True, graph=True)
    model.save_model(config.text_cnn_embedding_model, config.text_cnn_classifier_model)
    print('Auxiliary classifier model saved.')

    print('Testing on data/data_test.txt ...')
    model.load_model(config.text_cnn_embedding_model, config.text_cnn_classifier_model)
    model.test_on_dataset('data/data_test.txt')


if __name__ == '__main__':
    train_text_cnn()
