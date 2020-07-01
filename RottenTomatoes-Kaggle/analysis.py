from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Activation, LSTM
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

def get_stop_words():
    """
    加载停用词表
    """
    with open(r'D:\PycharmProjects\RottenTomatoes-Kaggle\stopwords.txt') as f:
        stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].replace("\'", '').strip()
    print("加载英文停用词表…………")
    return stopwords


def pre_process_data(total_text):
    """
    1.数据预处理
    将分词完毕的文本数据进行以下操作：
    （1）去除空格
    （2）去除标点符号
    （3）将英文全部转换为小写
    （4）去除停用词
    """
    total = []
    for i in total_text:
        text = i.lower()
        text = re.sub('[^a-zA-Z]', ' ', text)
        after_text = [j for j in text.strip().split('\t') if isinstance(j, str)]
        process_temp = ' '.join(after_text)
        stopwords = get_stop_words()
        final_text = [t for t in process_temp not in stopwords]
        total.append(final_text)

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

    return train_sequences, test_sequences, tokenizer


def n_tokens(train_content, test_content):
    """
    token长度
    """
    train_sequences, test_sequences, tokenizer = process_data(train_content, test_content)
    num_tokens = [len(tokens) for tokens in train_sequences]  # 获取所有token的长度
    num_tokens = np.array(num_tokens)
    print("tokens的长度：", len(num_tokens))
    print("平均tokens的长度：", np.mean(num_tokens))
    print("最长的评价tokens的长度：", np.max(num_tokens))
    print("最长的评价tokens的长度：", np.min(num_tokens))

    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    print(max_tokens)  # 19
    print(np.sum(num_tokens < max_tokens) / len(num_tokens))  # 取tokens的长度为19时，大约93%的样本被涵盖,0.9278610790721518

    return num_tokens


def show_trainData_tokens(train_content, test_content):
    """ 查看训练集长度 """
    num_tokens = n_tokens(train_content, test_content)
    plt.hist(num_tokens, bins=50)
    plt.xlabel("tokens的长度")
    plt.ylabel("tokens的数量")
    plt.title("tokens长度分布图")
    plt.show()


def process_train_sequences(train_label):
    """
    处理文本长度
    """
    train_sequences, test_sequences, tokenizer = process_data(train_content, test_content)
    train = sequence.pad_sequences(train_sequences, maxlen=48)
    test = sequence.pad_sequences(test_sequences, maxlen=48)  # 这里的长度可以是前面取到的max_tokens，也可以是它的最大长度，前者可以节约计算时间

    print("训练集特征数据维度：", train.shape)
    print("测试集特征数据维度：", test.shape)

    print("训练数据为：\n", train)
    # print("测试数据为：", test)

    train_label = to_categorical(train_label, 5)  # 5种情绪
    print("训练集标签维度：", train_label.shape)
    print("训练集标签：\n", train_label)

    return train, test, train_label, tokenizer


def split_data(train, train_label):
    """
    将训练样本进行切割，划分为训练集跟测试集
    """
    X_train, X_test, y_train, y_test = train_test_split(train, train_label, test_size=0.25, random_state=42)
    print("划分的训练集特征维度为：", X_train.shape)
    print("划分的测试集特征维度为：", X_test.shape)
    print("划分的训练集标签维度为：", y_train.shape)
    print("划分的测试集标签维度为：", y_test.shape)

    return X_train, X_test, y_train, y_test


def lstm_model(tokenizer):
    """
    LSTM 模型：通过 keras 搭建
    """
    max_features = len(tokenizer.index_word)  # 最多单词数
    print("max_features", max_features)
    # emb_dim = 128  # 128代表embedding层的向量维度
    #
    # model = Sequential()
    #
    # model.add(Embedding(max_features, emb_dim, mask_zero=True))
    # model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    # model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
    #
    # model.add(Dense(5, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    # print(model.summary())

    return model


def train_model(model, train, train_label):
    """
    训练模型
    """
    max_len = 48  # 此数要与前面padding时的长度一致，前面为48，此处也要为48
    epochs = 5  # 训练轮数
    batch_size = 80  # 这是指定批量的大小

    X_train, X_test, y_train, y_test = split_data(train, train_label)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))


def score_model(model):
    """
    评估模型
    """
    X_train, X_test, y_train, y_test = split_data(train, train_label)
    score = model.evaluate(X_test, y_test)
    print("LSTM模型损失率为：", score[0])
    print("LSTM模型得分为：", score[1])


def test_model(model, test):
    """
    测试模型
    """
    predict = model.predict_classes(test)
    # 预测测试集
    sample_data = pd.read_csv(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\sampleSubmission.csv', delimiter='\t')
    sample_data['Sentiment'] = predict
    sample_data.to_csv('lstm_pre.csv', index=False)


if __name__ == '__main__':
    train_content, train_label = train_dataset(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\train.tsv')
    test_content = test_dataset(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\test.tsv')
    # 展示标签类别分布
    show_label(train_label)
    # 展示文本长度
    show_trainData_tokens(train_content, test_content)
    train, test, train_label, tokenizer = process_train_sequences(train_label)
    # 训练样本划分为训练集和测试集
    split_data(train, train_label)
    # 构建LSTM模型
    model = lstm_model(tokenizer)
    # 训练LSTM模型
    # train_model(model, train, train_label)
    # # 评估LSTM模型
    # score_model(model)
    # # 测试LSTM模型
    # test_model(model, test)
