# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as functional
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config
from dataset import DualNovelDataSet


config = Config()


class LanguageModel(object):
    def __init__(self, direction):
        """初始化语言模型

        Args:
            direction: 代表语言模型是前向还是后向，原论文的程序有这个设定但本程序中没有用到
        """
        # 设定模型类型
        assert direction in config.lm_model_directions

        self.direction = direction
        self.model_path = config.lm_model_path.format(direction)

        # 设定数据集
        self.train_set = DualNovelDataSet(test=False, max_len=config.max_sentence_length, classification=False)
        self.test_set = DualNovelDataSet(test=True, max_len=config.max_sentence_length, classification=False)

        # 设定模型
        self.num_tokens = self.train_set.vocabulary.vocab_size
        self.pad = self.train_set.pad
        self.go = self.train_set.go
        self.eos = self.train_set.eos

        self.rnn_model = LanguageModelRNN(
            num_tokens=self.num_tokens,
            embedding_dim=config.lm_embedding_dim,
            hidden_dim=config.lm_hidden_dim,
            num_layers=config.lm_num_layers
        )
        if config.gpu:
            self.rnn_model = self.rnn_model.cuda()

        self.criterion = nn.CrossEntropyLoss()

        # 设置训练参数和优化器
        self.batch_size = config.lm_batch_size
        self.epochs = config.lm_epochs
        self.num_workers = config.lm_num_workers
        self.learning_rate = config.lm_learning_rate
        self.beta1 = config.lm_beta1
        self.beta2 = config.lm_beta2
        self.grad_norm_bound = config.lm_grad_norm_bound

        self.parameters = self.rnn_model.parameters()
        self.optimizer = Adam(self.parameters, self.learning_rate, (self.beta1, self.beta2))

    def train(self, verbose=False, graph=False):
        """训练模型

        Args:
            verbose: 是否在训练完每一个epoch后输出提示信息
            graph: 是否在训练完成后将损失函数和预测准确率变化绘制成图表
        """
        loss_list = []
        acc_list = []
        for epoch in range(self.epochs):
            epoch_loss = self.run_epoch(test=False)
            loss_list += epoch_loss
            if verbose:
                print('\n[TRAIN] Epoch {}, mean loss {}'.format(epoch, np.mean(epoch_loss)))

            train_accuracy, _ = self.run_epoch(test=True)
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
            plt.plot([x for x in range(self.epochs)], acc_list)
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
        self.rnn_model.train(train_mode)

    def run_epoch(self, test=False):
        """运行一个epoch

        Args:
            test: 布尔型，代表是否是测试模式

        Returns:
            test == True:
                mean_acc, mean_loss: 测试模式下，该epoch中的平均预测准确率和平均损失
            test == False:
                loss_list: 训练模式下，该epoch中各步的损失函数值
        """
        loss_list = []
        acc_list = []

        if not test:
            loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.rnn_model.train(mode=True)
        else:
            loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.rnn_model.train(mode=False)

        with tqdm(loader) as pbar:
            for data in pbar:
                # 读取数据
                inputs, targets = self.preprocess_data(data)

                # 生成循环神经网络使用的初始状态
                states = self.rnn_model.init_states(inputs.shape[0])
                if config.gpu:
                    states = [state.cuda() for state in states]
                states = tuple(self.detach(states))

                # 前向传播和计算损失
                outputs, states = self.rnn_model(inputs, states)
                loss = self.criterion(outputs, targets.reshape(-1))
                loss_list.append(loss.item())

                if not test:
                    # 训练模式，使用优化器进行更新
                    self.rnn_model.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.rnn_model.parameters(), self.grad_norm_bound)
                    self.optimizer.step()
                else:
                    # 测试模式，计算预测准确率
                    prediction = torch.argmax(outputs, dim=1)
                    label = targets.reshape(-1)

                    correct = (prediction == label).float()
                    null_mask = label.eq(self.train_set.pad).float()
                    valid_mask = 1.0 - null_mask
                    total_num = torch.sum(valid_mask).item()
                    masked_correct = correct * valid_mask
                    correct_num = torch.sum(masked_correct).item()

                    accuracy = correct_num / total_num
                    acc_list.append(accuracy)

        return (np.mean(acc_list), np.mean(loss_list)) if test else loss_list

    def inference(self, data_batch):
        """推断部分，输入语句（用编号表示），输出该句子出现的概率

        Args:
            data_batch: 形状为[batch_size, max_seq_len]的语句数据，其中存储的是单词的编号

        Returns:
            sentence_prob: 形状为[batch_size]的np.ndarray，表示每个语句的概率
        """
        with torch.no_grad():
            batch_size = data_batch.shape[0]

            states = self.rnn_model.init_states(batch_size)
            if config.gpu:
                states = [state.cuda() for state in states]
            states = tuple(state.detach() for state in states)

            # outputs shape: [batch_size * max_len, num_tokens]
            outputs, _ = self.rnn_model(data_batch, states)
            probability = functional.softmax(outputs, dim=1)
            probability = probability.view(batch_size, config.max_sentence_length, self.num_tokens)

            # null shape: [batch_size, max_len]
            null_mask = data_batch.eq(self.train_set.pad).int()

            sentence_prob = np.zeros(batch_size)
            for i in range(batch_size):
                prob = 1.0
                for j in range(1, config.max_sentence_length):
                    if null_mask[i, j] == 1:
                        break

                    word_id = data_batch[i, j]
                    prob *= probability[i, j - 1, word_id]

                sentence_prob[i] = prob

            return sentence_prob

    def detach(self, states):
        """将变量从当前计算图中分离，使之不需要梯度
        用于处理输入循环神经网络的初始状态

        Args:
            states: 一个列表或元组，每个元素代表一个初始状态

        Returns:
            states: 进行detach操作后的初始状态列表
        """
        return [state.detach() for state in states]

    def preprocess_data(self, data):
        """预处理数据，将正负样本数据进行合并，生成对应的数据和标签

        Args:
            data: DataLoader返回的数据，格式见本函数第一行

        Returns:
            inputs: 输入的语句数据，形状为[batch_size * 2, max_seq_len]的LongTensor
            targets: 输入的标签数据，状态同inputs
        """
        (input_0, target_0, len_0), (input_1, target_1, len_1) = data
        inputs = torch.cat([input_0, input_1], dim=0)
        targets = torch.cat([target_0, target_1], dim=0)

        if config.gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        return inputs, targets

    def save_model(self):
        """
        保存模型
        使用的文件路径在config中已经定义
        """
        torch.save(self.rnn_model.state_dict(), self.model_path)

    def load_model(self):
        """
        读取模型
        使用的文件路径在config中已经定义
        """
        self.rnn_model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))


class LanguageModelRNN(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_dim, num_layers):
        """初始化语言模型

        Args:
            num_tokens: 词典中单词的数量
            embedding_dim: Embedding层的维度
            hidden_dim: LSTM每个隐含状态的维度
            num_layers: LSTM的层数
        """
        super(LanguageModelRNN, self).__init__()

        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.num_tokens, self.embedding_dim)
        self.rnn_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_dim, num_tokens)
        # self.dense = nn.Sequential(
        #     nn.Linear(hidden_dim, num_tokens),
        #     nn.Softmax()
        # )

    def forward(self, inputs, hidden):
        """前向传播过程
        LSTM有两个隐含状态

        Args:
            inputs: 输入，形状为[batch_size, max_seq_len, num_tokens]
            hidden: 含两个元素的元组，每个元素为形状为[max_seq_len, batch_size, hidden_dim]的Tensor

        Returns:
            output: 输出结果，形状为[batch_size * max_seq_len, num_tokens]
            (hidden_h, hidden_c): LSTM部分最后的状态
        """
        embedded = self.embedding(inputs)

        output, (hidden_h, hidden_c) = self.rnn_lstm(embedded, hidden)
        # output shape: [batch_size * max_seq_len, hidden_dim]
        output = output.reshape(output.size(0) * output.size(1), output.size(2))

        # output shape: [batch_size * max_seq_len, num_tokens]
        output = self.dense(output)

        return output, (hidden_h, hidden_c)

    def init_states(self, batch_size):
        """生成可用于输入LSTM的隐含状态初值
        这里使用全零状态
        DataLoader在读取值数据集末尾时，可能会出现batch_size小于设定值的情况（剩余数据不足一个完整的batch）

        Args:
            batch_size: 一批数据中样本的数量
        """
        hidden_h = torch.zeros((self.num_layers, batch_size, self.hidden_dim))
        hidden_c = torch.zeros((self.num_layers, batch_size, self.hidden_dim))

        if config.gpu:
            hidden_h = hidden_h.cuda()
            hidden_c = hidden_c.cuda()

        return hidden_h, hidden_c


def train_language_model():
    print('Training language model...')
    model = LanguageModel(direction='forward')
    model.train(verbose=True, graph=True)
    model.save_model()
    print('Language model saved.')

    print('Loading language model...')
    model.load_model()

    print('Testing...')
    loader = DataLoader(model.train_set, batch_size=16)
    for data in loader:
        inputs, targets = model.preprocess_data(data)
        prob = model.inference(inputs)

        sentences = model.train_set.get_sentences(inputs)

        for i in range(len(sentences)):
            print('[{}], Prob: {}, Sentence: {}'.format(i, prob[i], ' '.join(sentences[i])))

        break


if __name__ == '__main__':
    train_language_model()
