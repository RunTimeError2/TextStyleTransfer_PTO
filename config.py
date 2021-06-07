# -*- coding: utf-8 -*-
"""
全局配置类
"""


class Config(object):
    def __init__(self):
        # 与数据有关
        self.style_0_data_path = './data/data1.txt'
        self.style_1_data_path = './data/data2.txt'
        self.vocab_file = './data/novels.vocab'

        self.embedding_dim = 64
        self.embedding_file = None

        self.max_sentence_length = 30

        self.gpu = False

        self.temp_att = 0.05
        self.sample_temperature = 1

        # 与TextCNN风格分类器有关
        self.text_cnn_kernels = [1, 2, 3, 4, 5]
        self.text_cnn_channels = 48
        self.text_cnn_dropout_rate = 0.5

        self.text_cnn_learning_rate = 1e-2
        self.text_cnn_beta1 = 0.5
        self.text_cnn_beta2 = 0.999

        self.text_cnn_batch_size = 16
        self.text_cnn_epochs = 20
        self.text_cnn_num_workers = 4

        self.text_cnn_embedding_model = './saved_models/text_cnn_embedding.ckpt'
        self.text_cnn_classifier_model = './saved_models/text_cnn_classifier.ckpt'

        # 与Pointer(用AttentionClassifier实现)有关
        self.pointer_learning_rate = 1e-2
        self.pointer_beta1 = 0.5
        self.pointer_beta2 = 0.5

        self.pointer_batch_size = 16
        self.pointer_epochs = 10
        self.pointer_num_workers = 4

        self.pointer_embedding_model = './saved_models/pointer_embedding.ckpt'
        self.pointer_att_classifier_model = './saved_models/pointer_att_classifier.ckpt'

        self.hidden_dim = 192
        self.num_layers = 1
        self.bidirectional = True
        self.dropout_rate = 0.5

        # 与语言模型有关
        self.lm_model_path = './saved_models/language_model_{}.ckpt'
        self.lm_model_directions = ['forward', 'backward']

        self.lm_hidden_dim = 256
        self.lm_embedding_dim = 128
        self.lm_num_layers = 3
        self.lm_num_workers = 4

        self.lm_batch_size = 16
        self.lm_epochs = 25
        self.lm_learning_rate = 1e-2
        self.lm_beta1 = 0.5
        self.lm_beta2 = 0.999
        self.lm_grad_norm_bound = 0.5

        # 与自编码器有关
        self.encoder_model_path = './saved_models/ae_encoder.ckpt'
        self.encoder_num_layers = 2
        self.encoder_bidirectional = True
        self.encoder_num_workers = 4

        self.decoder_model_path = './saved_models/ae_decoder.ckpt'
        self.decoder_num_layers = 2
        self.decoder_bidirectional = True
        self.decoder_num_workers = 4

        self.ae_batch_size = 16
        self.ae_epochs = 50
        self.ae_num_workers = 4
        self.ae_learning_rate = 1e-2
        self.ae_beta1 = 0.5
        self.ae_beta2 = 0.999

        # 与PTO和Operator有关
        self.operator_hidden_dim = 128
        self.operator_num_layers = 3
        self.operator_dropout_rate = 0.2
        self.operator_bidirectional = True
        self.operator_random_sample = True

        self.operator_learning_rate = 1e-3
        self.operator_beta1 = 0.5
        self.operator_beta2 = 0.999

        self.pto_batch_size = 8
        self.pto_epochs = 10
        self.pto_num_workers = 4
        self.pto_clip_norm = float('inf')

        self.pto_iterations = 10
        self.pto_model_path_template = './saved_models/pto_{}_{}.ckpt'

        self.update_data_thresh = 0.15

        self.total_class_loss_coef = 1.0
        self.edit_class_loss_coef = 1.0
        self.edit_lm_loss_coef = 0.8
        self.semantic_loss_coef = 0.5
