# 基于Point-Then-Operate方法的迭代写作风格迁移
**2021春季学期模式识别大作业**
*写作风格迁移任务*

## 主要参考文献

本项目主要参考了以下文献

> C. Wu, X. Ren, F. Luo and X. Sun, 
"A Hierarchical Reinforced Sequence Operation Method for Unsupervised Text Style Transfer," 
Proceedings of the 57th Conference of the Association for Computational Linguistics (ACL),
vol. 1, pp. 4873-4883, 2019.

与原论文有关的资料如下
[原论文](https://www.aclweb.org/anthology/P19-1482)
|[原论文slides](https://github.com/ChenWu98/Point-Then-Operate/blob/master/static/ACL_2019_PTO_Slides.pdf)
|[原论文代码](https://github.com/ChenWu98/Point-Then-Operate)

## 程序项目结构

### data目录

储存了模型训练所使用的数据集，其中txt文件为原始的以及处理过的文本数据，
novels.vocab是程序所使用的字典数据

### saved_models目录

储存了项目中各个模型的训练结果，具体目录设置见config.py文件

### 根目录

因本项目中程序文件较少，因此全部放在根目录。各程序文件的内容如下

- adaptive_dataset.py: 实现了可在训练过程中进行扩充的数据集
- attention_visualize.py: 用于对语句以及对应的Attention权重进行可视化。未在主程序中直接调用，仅作为写报告时的辅助程序
- autoencoder.py: Seq2Seq之编码器，用于对语句的语义进行表示。文件中自带预训练的代码
- classifier.py: 基于TextCNN的文本风格分类器，文件中自带预训练的代码
- config.py: 全局配置程序，所有超参数、路径等配置均在此文件中进行定义
- dataset.py: 基于pytorch的DataSet接口实现了数据集
- language_model.py: 实现了语言模型，可用于判断语句通顺程度。文件中自带预训练的代码
- load_text.py: 读取原始文本数据并生成相应的字典对象
- operators.py: PTO算法中的各种Operator模型，无法预训练，在运行主程序时将会被调用
- pointer.py: Pointer模型，是一个带有Attention机制的分类器。文件中自带预训练的代码
- pto_main.py: PTO主模型。文件中带有训练和测试的代码
- train.py: 训练入口程序，运行此程序将会自动地逐个训练每一个模型

## 运行方式

详细的程序项目结构见[报告](slides.pdf)。其中各模型都是可以单独训练的，训练方式如下

- 训练辅助分类器模型：执行python classifier.py
- 训练Pointer模型：执行python pointer.py
- 训练语言模型：执行python language_model.py
- 训练自编码器模型：执行python autoencoder.py
- 训练PTO主模型：执行python pto_main.py

*各程序都没有额外的参数*

因为saved_models下已经保存了各辅助模型的参数，因此只执行pto_main.py来训练主模型也是可以的。
此外，执行python train.py可以自动地逐个训练各个模型。

训练完成后，程序会自动显示训练过程中的损失函数变化与训练过程中的准确率（仅部分模型会显示准确率），并进行测试（如果可以测试的话）。
PTO主模型会输出数据集中部分样本的迁移结果，作为测试结果

因为数据集和模型规模有限，不需要额外下载数据和模型参数

## 运行环境的问题

程序支持GPU下运行，修改config.py中self.gpu参数可以选择是否在GPU环境下运行。
但因程序是在CPU环境下编写并完成测试的，根据我们在文献复现过程中的经验，在GPU环境下运行可能出现因PyTorch版本导致的问题，
如果遇到问题请切换至CPU环境运行。
