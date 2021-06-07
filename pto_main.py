# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import Config
from dataset import sample_2d, DualNovelDataSet, Vocabulary
from adaptive_dataset import AdaptiveDataSet
from pointer import PointerModule
from classifier import AuxiliaryStyleClassifier
from autoencoder import SeqAutoEncoder
from language_model import LanguageModel
from operators import Delete, InsertBehind, InsertFront, Replace


config = Config()


class PTOMain(object):
    """
    实现Point-Then-Operate算法的类
    包括模型建立、训练和推断步骤
    """

    def __init__(self, verbose=False):
        """初始化类
        包括读取数据集、读取预训练模型、建立模型和设定参数

        Args:
            verbose: 初始化过程中是否输出提示信息
        """
        # 建立数据集
        if verbose:
            print('Loading data set...')
        self.train_set = AdaptiveDataSet()
        self.test_set = AdaptiveDataSet()

        self.vocabulary = Vocabulary(config.vocab_file)

        # 读取预训练模型，都不再进行训练
        if verbose:
            print('Loading pre-trained models...')

        self.auxiliary_classifier = AuxiliaryStyleClassifier()
        self.auxiliary_classifier.load_model()
        self.auxiliary_classifier.set_training(False)

        self.pointer = PointerModule()
        self.pointer.load_model()
        self.pointer.set_training(False)

        self.language_model = LanguageModel(direction='forward')
        self.language_model.load_model()
        self.language_model.set_training(False)

        self.autoencoder = SeqAutoEncoder()
        self.autoencoder.load_model()
        self.autoencoder.set_training(False)

        # 建立需要训练的模型
        if verbose:
            print('Building models...')

        # Delete模块因为不含参数，只需要一个
        self.delete = Delete(pad_id=self.train_set.pad)

        # 其他模块都需要两个，对应将每种风格迁移到另一种风格
        self.insert_front = [
            InsertFront(
                hidden_dim=config.operator_hidden_dim,
                num_layers=config.operator_num_layers,
                dropout_rate=config.operator_dropout_rate,
                bidirectional=config.operator_bidirectional,
                random_sample=config.operator_random_sample
            ),
            InsertFront(
                hidden_dim=config.operator_hidden_dim,
                num_layers=config.operator_num_layers,
                dropout_rate=config.operator_dropout_rate,
                bidirectional=config.operator_bidirectional,
                random_sample=config.operator_random_sample
            )
        ]

        self.insert_behind = [
            InsertBehind(
                hidden_dim=config.operator_hidden_dim,
                num_layers=config.operator_num_layers,
                dropout_rate=config.operator_dropout_rate,
                bidirectional=config.operator_bidirectional,
                random_sample=config.operator_random_sample
            ),
            InsertBehind(
                hidden_dim=config.operator_hidden_dim,
                num_layers=config.operator_num_layers,
                dropout_rate=config.operator_dropout_rate,
                bidirectional=config.operator_bidirectional,
                random_sample=config.operator_random_sample
            )
        ]

        self.replace = [
            Replace(
                hidden_dim=config.operator_hidden_dim,
                num_layers=config.operator_num_layers,
                dropout_rate=config.operator_dropout_rate,
                bidirectional=config.operator_bidirectional,
                random_sample=config.operator_random_sample
            ),
            Replace(
                hidden_dim=config.operator_hidden_dim,
                num_layers=config.operator_num_layers,
                dropout_rate=config.operator_dropout_rate,
                bidirectional=config.operator_bidirectional,
                random_sample=config.operator_random_sample
            )
        ]

        self.operators = [self.insert_front, self.insert_behind, self.replace]

        # 设置优化器
        if verbose:
            print('Setting optimizer and parameters...')

        self.operator_trainable_variables = []
        for operator_list in self.operators:
            for operator in operator_list:
                for k, v in operator.state_dict(keep_vars=True).items():
                    if v.requires_grad:
                        self.operator_trainable_variables.append(v)
        self.operator_optimizer = Adam(
            self.operator_trainable_variables,
            config.operator_learning_rate,
            (config.operator_beta1, config.operator_beta2)
        )

        # 训练所需参数
        self.batch_size = config.pto_batch_size
        self.epochs = config.pto_epochs
        self.num_workers = config.pto_num_workers

        self.loss_criterion = WeightedLossCriterion()

    def save_model(self):
        """
        将模型保存到指定路径
        """
        for model_i, model_name in enumerate(['ins_front', 'ins_behind', 'replace']):
            for style in [0, 1]:
                model_path = config.pto_model_path_template.format(model_name, style)
                torch.save(self.operators[model_i][style].state_dict(), model_path)

    def load_model(self):
        """
        从指定路径的文件读取模型参数
        """
        for model_i, model_name in enumerate(['ins_front', 'ins_behind', 'replace']):
            for style in [0, 1]:
                model_path = config.pto_model_path_template.format(model_name, style)
                self.operators[model_i][style].load_state_dict(
                    torch.load(model_path, map_location=lambda storage, loc: storage)
                )

    def train(self, verbose=False, graph=False, add_data=True):
        """训练模型
        允许将训练过程中生成的数据重新添加进数据集

        Args:
            verbose: 每个Epoch训练完成后，是否输出提示信息
            graph: 全部训练完成后，是否绘制损失函数值变化图像
            add_data: 训练过程中，是否将生成的新数据加入数据集
        """
        total_loss_list = []

        for epoch in range(self.epochs):
            if verbose:
                print('[TRAIN] Epoch {}, length of data set = {}'.format(epoch, self.train_set.__len__()))

            loss_list, (adding_sentences, adding_length, adding_style, adding_iter) = self.train_epoch()

            total_loss_list += loss_list

            if add_data:
                self.train_set.add_data(adding_sentences, adding_length, adding_style, adding_iter)
                self.train_set.current_iter += 1

            if verbose:
                print('[TRAIN] Epoch {}, mean loss = {}'.format(epoch, np.mean(loss_list)))

        if graph:
            plt.figure()
            plt.plot([x for x in range(len(total_loss_list))], total_loss_list)
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.title('Training loss')
            plt.grid()
            plt.show()

    def set_training(self, train_mode):
        """设定训练/测试模式

        Args:
            train_mode: 布尔型，是否是训练模式
        """
        for operator_list in self.operators:
            for operator in operator_list:
                operator.train(mode=train_mode)

    def train_epoch(self):
        """训练一个epoch
        步骤包括读取数据、执行前向传播、计算梯度和更新参数、更新数据集

        Returns:
            loss_list: 训练过程中损失函数列表，一个数值对应一个step
            (adding_sentences, adding_length, adding_style, adding_iter): 要添加进数据集的新数据，包括修改过的语句、
                                                                          语句的有效长度、语句对应的风格、语句经过的迭代次数
        """
        loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.set_training(True)

        # 储存要添加进数据集的数据
        adding_sentences = []
        adding_length = []
        adding_style = []
        adding_iter = []

        loss_list = []

        with tqdm(loader) as pbar:
            for data in pbar:
                # 读取数据并计算null_mask
                (sentence_0, len_0, iter_0), (sentence_1, len_1, iter_1) = data
                null_mask_0 = sentence_0.eq(self.train_set.pad)
                null_mask_1 = sentence_1.eq(self.train_set.pad)

                # 针对不同风格的数据，分别执行前向操作
                loss_0, (add_sentence_0, add_len_0, add_style_0, add_iter_0) = self.forward_pto(
                    sentence_0, len_0, null_mask_0, iter_0, 0
                )
                loss_1, (add_sentence_1, add_len_1, add_style_1, add_iter_1) = self.forward_pto(
                    sentence_1, len_1, null_mask_1, iter_1, 1
                )

                # 计算损失函数、限制梯度并执行参数更新步骤
                total_loss = loss_0 + loss_1
                self.operator_optimizer.zero_grad()
                total_loss.backward()
                if config.pto_clip_norm < float('inf'):
                    clip_grad_norm_(self.operator_trainable_variables, max_norm=config.pto_clip_norm)
                self.operator_optimizer.step()

                loss_list.append(total_loss.item())

                # 储存要添加进数据集的数据
                adding_sentences += add_sentence_0 + add_sentence_1
                adding_length += add_len_0 + add_len_1
                adding_style += add_style_0 + add_style_1
                adding_iter += add_iter_0 + add_iter_1

        adding_sentences = torch.stack(adding_sentences)
        adding_length = torch.stack(adding_length)
        adding_style = torch.Tensor(adding_style)
        adding_iter = torch.stack(adding_iter)

        return loss_list, (adding_sentences, adding_length, adding_style, adding_iter)

    def forward_pto(self, sentence, length, null_mask, iter, style):
        """前向传播步骤
        统一了不同风格数据的前向传播

        Args:
            sentence: 需要修改的语句，是形状为[batch_size, max_seq_len]的LongTensor
            length: 语句对应的长度，是形状为[batch_size]的Tensor
            null_mask: 语句对应的填充符蒙版，形状同sentence
            iter: 当前的迭代步数，非负整数
            style: 当前使用的样本对应的风格，取0或1

        Returns:
            total_loss: 该批数据对应的损失函数值
            (adding_sentences, adding_length, adding_style, adding_iter): 由这一批数据生成的、可以添加进数据集的新数据
        """
        batch_size = sentence.shape[0]

        class_probs, class_logits, attention_weights = self.pointer.inference(sentence, length, null_mask)
        class_probs = class_probs.detach()

        # edit_positions代表各语句需要修改的位置，是形状为[batch_size]的Tensor
        edit_positions, position_prob = sample_2d(
            probability=attention_weights,
            temperature=config.sample_temperature
        )

        # 将各种操作都进行尝试，得到相应的结果
        sentence_del, length_del = self.delete(sentence, edit_positions, length)
        _, probs_ins_front, sentence_ins_front, length_ins_front = self.insert_front[style](
            sentence, edit_positions, length
        )
        _, probs_ins_behind, sentence_ins_behind, length_ins_behind = self.insert_behind[style](
            sentence, edit_positions, length
        )
        _, probs_replace, sentence_replace, length_replace = self.replace[style](
            sentence, edit_positions, length
        )

        edit_flag = [True] * (batch_size * 4) + [False] * batch_size
        sentence_edited = torch.cat(
            [sentence_del, sentence_ins_front, sentence_ins_behind, sentence_replace, sentence],
            dim=0
        )
        length_edited = torch.cat(
            [length_del, length_ins_front, length_ins_behind, length_replace, length],
            dim=0
        )
        null_mask_edited = sentence_edited.eq(self.train_set.pad)
        total_iter = torch.cat([iter] * 5, dim=0)

        # 对语句修改的结果进行评判，计算相应的损失/回报函数
        class_probs_expanded = torch.cat([class_probs] * 5, dim=0)

        # 风格极性/分类结果的回报/损失
        class_edit_probs, _ = self.auxiliary_classifier.inference(sentence_edited)
        class_edit_probs = class_edit_probs.detach()
        # 要求修改后的风格与修改前相比越大（越接近1）越好
        class_loss = 1.0 - torch.abs(class_edit_probs - class_probs_expanded)
        # 对有单词改动的句子（插入和替换）单独计算
        edit_sample_probs = torch.cat([probs_ins_front, probs_ins_behind, probs_replace], dim=0)
        # 计算与分类结果有关的回报，结果是标量
        total_class_loss = self.loss_criterion(
            sample_probs=class_probs_expanded,
            losses=class_loss
        )
        edit_class_loss = self.loss_criterion(
            sample_probs=edit_sample_probs,
            losses=class_loss[batch_size: batch_size * 4]
        )

        # 对经过修改的句子单独计算语言模型输出的概率，越接近1越好
        lm_edit_probs = self.language_model.inference(sentence_edited)
        lm_loss = torch.Tensor(1.0 - lm_edit_probs)
        edit_lm_loss = self.loss_criterion(
            sample_probs=edit_sample_probs,
            losses=lm_loss[batch_size: batch_size * 4]
        )

        # 对经过修改的句子分别计算语义损失
        semantic_loss_del = self.autoencoder.mean_difference(sentence_del, sentence)
        semantic_loss_ins_front = self.autoencoder.mean_difference(sentence_ins_front, sentence)
        semantic_loss_ins_behind = self.autoencoder.mean_difference(sentence_ins_behind, sentence)
        semantic_loss_replace = self.autoencoder.mean_difference(sentence_replace, sentence)

        semantic_loss = semantic_loss_del + semantic_loss_ins_front + semantic_loss_ins_behind + semantic_loss_replace

        # 加权计算总的回报
        total_loss = total_class_loss * config.total_class_loss_coef + \
            edit_class_loss * config.edit_class_loss_coef + \
            edit_lm_loss * config.edit_lm_loss_coef + \
            semantic_loss * config.semantic_loss_coef

        # 将修改后的语句加入数据集，要求修改后语句的风格类别概率变化要大于指定阈值
        adding_sentences = []
        adding_length = []
        adding_style = []
        adding_iter = []

        # 如果改动对类别的影响足够大（超过指定阈值），则认为是可以加入数据集的新数据
        confidence_diff = torch.abs(class_edit_probs - class_probs_expanded)
        for i in range(sentence_edited.shape[0]):
            if edit_flag[i] and confidence_diff[i].item() > config.update_data_thresh:
                adding_sentences.append(sentence_edited[i])
                adding_length.append(length_edited[i])
                adding_style.append(style)
                adding_iter.append(total_iter[i])

        return total_loss, (adding_sentences, adding_length, adding_style, adding_iter)

    def inference(self, sentences, length):
        """推断步骤
        输入一批语句，利用模型将其转换为另一种风格

        Args:
            sentences: 需要转换风格的语句，不一定要是同一种风格，是形状为[batch_size, max_seq_len]的Tensor
            length: 各语句的有效长度，是形状为[batch_size]的Tensor

        Returns:
            sentences_transfer: 转换风格后的语句，形状同输入
            length_transfer: 转换风格后的语句对应的有效长度，形状同输入
        """
        sentences_transfer, length_transfer = [], []
        for i in range(sentences.shape[0]):
            sentence_i, length_i = sentences[i], length[i]
            class_prob, _ = self.auxiliary_classifier.inference(sentence_i.unsqueeze(0))
            style = 0 if class_prob[0].item() < 0.5 else 1

            sentence_edit, length_edit = self.style_transfer(
                sentence_i.unsqueeze(0), length_i.unsqueeze(0), style, config.pto_iterations
            )

            sentences_transfer.append(sentence_edit[0])
            length_transfer.append(length_edit[0])

        sentences_transfer = torch.stack(sentences_transfer, dim=0)
        length_transfer = torch.stack(length_transfer, dim=0)

        return sentences_transfer, length_transfer

    def style_transfer(self, sentences, length, style, iterations):
        """对单一风格的样本的推断步骤
        将指定风格的样本转换为另一种风格

        Args:
            sentences: 需要转换风格的语句，是形状为[batch_size, max_seq_len]的Tensor
            length: 需要转换风格的语句的有效长度，是形状为[batch_size]的Tensor
            style: 当前语句的风格
            iterations: 转换风格过程中迭代的次数

        Returns:
            sentences: 转换风格后的语句，形状同输入
            length: 转换风格后语句的有效长度，形状同输入
        """
        batch_size = sentences.shape[0]

        with torch.no_grad():
            for step in range(iterations):
                null_mask = sentences.eq(self.train_set.pad)

                # 使用Pointer计算需要修改的位置
                _, _, attention_weights = self.pointer.inference(sentences, length, null_mask)
                edit_positions = torch.argmax(attention_weights, dim=1)  # 推断过程不再随机采样

                # 使用各种方法对语句进行编辑
                sentence_del, length_del = self.delete(sentences, edit_positions, length)
                _, _, sentence_ins_front, length_ins_front = self.insert_front[style](
                    sentences, edit_positions, length
                )
                _, _, sentence_ins_behind, length_ins_behind = self.insert_front[style](
                    sentences, edit_positions, length
                )
                _, _, sentence_replace, length_replace = self.replace[style](
                    sentences, edit_positions, length
                )

                generated_sentences = [sentence_del, sentence_ins_front, sentence_ins_behind, sentence_replace, sentences]
                updated_length = [length_del, length_ins_front, length_ins_behind, length_replace, length]
                style_difference = np.zeros((batch_size, 5))

                # 对编辑后的语句，计算其类别（概率），找出并保留效果最好的
                for i, sentence_edited in enumerate(generated_sentences):
                    class_edit_probs, _ = self.auxiliary_classifier.inference(sentence_edited)
                    for j in range(batch_size):
                        style_difference[j, i] = abs(style - class_edit_probs[j].item())

                best_index = np.argmax(style_difference, axis=1)
                best_sentences = []
                best_length = []
                for i in range(batch_size):
                    best_sentences.append(generated_sentences[best_index[i]][i])
                    best_length.append(updated_length[best_index[i]][i])
                best_sentences = torch.stack(best_sentences, dim=0)
                best_length = torch.stack(best_length, dim=0)

                # 准备下一次迭代
                sentences = best_sentences
                length = best_length

        return sentences, length


class WeightedLossCriterion(nn.Module):
    """
    加权损失函数计算，以采样的概率为权重，概率越低权重越大
    """

    def __init__(self):
        super(WeightedLossCriterion, self).__init__()

    def forward(self, sample_probs, losses, mask=None):
        """加权损失函数计算过程

        Args:
            sample_probs: 采样概率
            losses: 损失函数值，需要和采样概率同形状
            mask: 蒙版（可选），需要和改样概率、损失函数值同形状

        Returns:
            output: 一个标量值，表示损失函数值的加权平均值
        """
        if sample_probs is None:
            zero_rewards = torch.zeros([1]).squeeze(0)
            if config.gpu:
                zero_rewards = zero_rewards.cuda()
            return zero_rewards

        sample_probs = sample_probs.contiguous().view(-1)
        sample_log_probs = torch.log(sample_probs)
        losses = losses.contiguous().view(-1)

        if mask is not None:
            mask = mask.float().contiguous().view(-1)
            output = -sample_log_probs * losses * mask
            output = torch.sum(output) / torch.sum(mask)
        else:
            output = -sample_log_probs * losses
            output = output.mean()

        return output


def main():
    print('Training main model of PTO...')
    pto_main = PTOMain(verbose=True)
    pto_main.train(verbose=True, graph=True)
    pto_main.save_model()
    print('Main model of PTO saved.')

    print('Loading main model of PTO...')
    pto_main.load_model()

    print('Testing...')
    loader = DataLoader(pto_main.train_set, batch_size=8, shuffle=True, num_workers=4)
    for batch_i, data in enumerate(loader):
        (sentence_0, len_0, _), (sentence_1, len_1, _) = data
        sentence = torch.cat([sentence_0, sentence_1], dim=0)
        length = torch.cat([len_0, len_1], dim=0)
        styles = [0] * sentence_0.shape[0] + [1] * sentence_0.shape[0]

        sentence_transfer, length_transfer = pto_main.inference(sentence, length)

        text_origin = pto_main.train_set.get_sentence(sentence, length)
        text_transfer = pto_main.train_set.get_sentence(sentence_transfer, length_transfer)

        for i in range(sentence.shape[0]):
            print('Batch {}, Sentence {}:'.format(batch_i, i))
            print('[Original] style{}: {}'.format(styles[i], ' '.join(text_origin[i])))
            print('    [Transferred]: {}'.format(' '.join(text_transfer[i])))
            print('')
        break  # 这里只使用了一个batch的数据进行测试，可以按需增加


if __name__ == '__main__':
    main()
