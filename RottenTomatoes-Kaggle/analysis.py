from tensorflow.keras.preprocessing.text import Tokenizer
from pylab import mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

mpl.rcParams['font.sans-serif'] = ['SimHei']

"""
    多分类：
    0 - negative
    1 - somewhat negative
    2 - neutral
    3 - somewhat positive
    4 - positive
    """


def pre_process_data(total_text):
    """
    1.数据预处理
    将分词完毕的文本数据进行以下操作：
    （1）去除空格
    （2）去除标点符号
    （3）将英文全部转换为小写
    """
    total = []
    for i in total_text:
        text = i.lower()
        text = re.sub('[^a-zA-Z]', '', text)
        after_text = [j for j in text.strip().split('\t') if isinstance(j, str)]
        process_temp = ''.join(after_text)
        total.append(process_temp)

    return total


def train_dataset(filename):
    """ 读取训练集 """
    train_data = pd.read_csv(filename, delimiter='\t')
    train_content = train_data['Phrase']
    train_content = pre_process_data(train_content)  # 处理训练集文本
    train_label = train_data['Sentiment']  # 获取训练集标签

    print("训练集特征长度：", len(train_content))  # 156060
    print("训练集标签长度：", len(train_label))  # 156060

    return train_content, train_label


def test_dataset(filename):
    """ 读取测试集 """
    test_data = pd.read_csv(filename, delimiter='\t')
    test_content = test_data['Phrase']  # 处理测试集文本
    test_content = pre_process_data(test_content)  # 获取测试集标签

    print("测试集特征长度：", len(test_content))  # 66292

    return test_content


def show_label(train_label):
    """
    查看评分的类别分布
    """
    labels = set(train_label)  # 计算不重复类别数目
    x = []
    y = []
    for i in labels:
        x.append(i)
        y.append(train_label[train_label == i].size)
    plt.figure(111)
    plt.bar(x, y)
    plt.xlabel("Label--标签/评分类别")
    plt.ylabel("count--总数")
    plt.show()


def process_data(train_content, test_content):
    """
    Tokenizer：一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_content)
    train_sequences = tokenizer.texts_to_sequences(train_content)
    test_sequences = tokenizer.texts_to_sequences(test_content)

    num_tokens = [len(tokens) for tokens in train_sequences]  # 获取所有token的长度
    num_tokens = np.array(num_tokens)
    print("tokens的长度：", len(num_tokens))
    print("平均tokens的长度：", np.mean(num_tokens))
    print("最长的评价tokens的长度：", np.max(num_tokens))
    print("最长的评价tokens的长度：", np.min(num_tokens))

    return num_tokens


def show_trainData_tokens(train_content, test_content):
    """ 查看训练集长度 """
    num_tokens = process_data(train_content, test_content)
    plt.hist(num_tokens, bins=50)
    plt.xlabel("tokens的长度")
    plt.ylabel("tokens的数量")
    plt.title("tokens长度分布图")
    plt.show()


if __name__ == '__main__':
    train_content, train_label = train_dataset(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\train.tsv')
    test_content = test_dataset(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\test.tsv')
    show_label(train_label)
    show_trainData_tokens(train_content, test_content)
    # process_data(train_content, test_content)\