# -*- coding: utf-8 -*-
"""
用于读取文本数据并进行预处理的程序
"""
from tqdm import tqdm
import os
import pickle
from collections import Counter


# 需要替换的特殊单词/缩写等，可能会影响到标点
special_replacements = [
    ['St. Louis', 'StLouis'],
    ['i\'m', 'i am'],
    ['\'re', ' are'],
    ['ain\'t', 'are not'],
    ['hain\'t', 'has not'],
    ['warn\'t', 'was not'],
    # ['']
]
# 需要去掉的标点符号
punctuation = [',', '--', ':', ';', '<', '>', '/', '(', ')', '[', ']', '{', '}']
# 代表语句结束的标点符号
endding_punctuation = ['.', '?', '!', '"']


def delete_double_spaces(text):
    """删除连续的空格

    Args:
        text: 字符串
    
    Returns:
        text: 删除连续空格后的字符串，其中只会出现单个的空格
    """
    while text.find('  ') != -1:
        text = text.replace('  ', ' ')
    return text


def split_with_multiple_separator(sentence, separators):
    """实现使用多个分隔符来分割语句
    因为代表语句结束的标点中含有'?'等特殊符号，会干扰正则的使用

    Args:
        sentence: 字符串，待分割的语句
        separators: 元素为字符串的list，代表语句结束的标点
    
    Returns:
        sentences: 元素为字符串的list，代表分隔后的语句
    """
    sentences = [sentence]
    for separator in separators:
        sentence_list = []
        for item in sentences:
            sentence_list += item.split(separator)
        sentences = [item.strip() for item in sentence_list if item.strip() != '']

    return sentences


def load_text_file(file_path, min_sentence_length=4, save_path=None):
    """从txt文件中读取信息

    Args:
        filename: 文件的路径
        min_sentence_length: 最小的语句长度，少于这个长度的句子将被丢弃
        save_path: 如果有效，则表示将读取的数据保存到指定文件，文本格式
    
    Returns:
        sentence_list: 一个list, 每个元素是一个list代表单个句子，其元素为字符串，代表单词
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    all_text = ' '.join(lines)
    all_text = all_text.lower()

    for src, dest in special_replacements:
        all_text = all_text.replace(src, dest)

    sentences = split_with_multiple_separator(all_text, endding_punctuation)

    sentence_list = []
    for item in sentences:
        sentence = item
        for punc in punctuation:
            sentence = sentence.replace(punc, ' ')
        sentence = delete_double_spaces(sentence)
        sentence = sentence.strip()
        sentence = sentence.split(' ')
        if len(sentence) >= min_sentence_length:
            sentence_list.append(sentence)
        
    if save_path:
        with open(save_path, 'w') as f:
            for sentence in sentence_list:
                f.write(' '.join(sentence) + '\n')

    return sentence_list


def update_vocab(data, load_path, save_path, min_occur=2):
    """更新或创建字典，并保存到文件

    Args:
        data: 保存所有单词的列表，格式同load_text_file返回值
        load_path: 现有词典文件所在路径，如果为None则不读取
        save_path: 要保存的路径，如果为None则不保存
        min_occur: 最小词频，词频小于这个值的单词将被删去
    
    Returns:
        vocab_size: 词典中单词数量
        word2id: 字典，将单词转换为编号
        id2word: 字典，将编号转换为单词
    """
    if load_path and os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            size, word2id, id2word = pickle.load(f)
    else:
        id2word = ['<pad>', '<go>', '<eos>', '<unk>']
        word2id = {tok: ix for ix, tok in enumerate(id2word)}
    
    words = [word for sentence in data for word in sentence]
    cnt = Counter(words)
    for word in cnt:
        if cnt[word] >= min_occur and word not in word2id.keys():
            word2id[word] = len(word2id)
            id2word.append(word)
    vocab_size = len(word2id)

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((vocab_size, word2id, id2word), f, pickle.HIGHEST_PROTOCOL)
    
    return vocab_size, word2id, id2word


def main():
    sentence_list = load_text_file('data/The Tragedy of Pudd\'nhead Wilson.txt', save_path='data/data1.txt')
    sentence_list += load_text_file('data/Metzengerstein.txt', save_path='data/data2.txt')

    vocab_size, word2id, id2word = update_vocab(sentence_list, None, save_path='data/novels.vocab')
    print('Size of dictionary: {}'.format(vocab_size))

    # sentence_list = load_text_file('data/test-Pudd\'nhead.txt', save_path='data/data_test.txt')


if __name__ == '__main__':
    main()
