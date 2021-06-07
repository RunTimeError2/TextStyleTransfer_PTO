# -*- coding: utf-8 -*-
from classifier import train_text_cnn
from pointer import pre_train_pointer
from language_model import train_language_model
from autoencoder import train_autoencoder
from pto_main import main as pto_main
from config import Config


config = Config()


def main():
    # 训练三个辅助模型
    train_text_cnn()
    pre_train_pointer()
    train_language_model()
    train_autoencoder()
    # 训练主模型
    pto_main()


if __name__ == '__main__':
    main()
