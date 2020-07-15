from nltk.corpus import stopwords
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras import optimizers
import pandas as pd
import numpy as np
import spacy

nlp = spacy.load('en_core_web_md')  # 加载预训练模型


def get_stop_words():
    """
    加载停用词表
    """
    with open('stopwords.txt') as f:
        stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].replace("\'", '').strip()
    print("加载英文停用词表…………")
    return stopwords


stopwords = get_stop_words()  # 获取停用词表


def creat_dataset():
    """
    加载数据集
    :train_data, test_data:
    """
    train_data = pd.read_csv(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\train.tsv', delimiter='\t')
    test_data = pd.read_csv(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\test.tsv', delimiter='\t')
    print("NLP以及数据集已加载…………")

    return train_data, test_data


def process_dataset():
    """
    处理数据集：
    1.合并原始数据集（训练集、测试集）
    2.选取数据集中用户完整评论（id）
    :text, sentences:
    """
    train_data, test_data = creat_dataset()
    text = pd.concat([train_data.Phrase, test_data.Phrase], axis=0)  # 合并训练集、测试集特征
    combined = pd.concat([train_data, test_data], axis=0)  # 合并训练集、测试集
    # 根据数据集特点，选取数据集中每位用户第一条评论（即完整的评论）
    sentences = []
    for i in combined.SentenceId.unique():
        sentences.append(combined.loc[combined.SentenceId == i]['Phrase'].iloc[0])

    return text, sentences


def preprocess(x):
    print("x:", x)
    x = nlp(x.lower())  # 大小写
    tokens = [t for t in x]
    tokens = [t for t in tokens if t.text not in stopwords]  # 去除停用词
    tokens = [t for t in tokens if t.is_punct == False]
    tokens = [t for t in tokens if len(t) >= 3]  # 取单词长度大于等于3的，此举可以去除标点符号
    tokens = [t.lemma_ for t in tokens]  # 词性还原
    return tokens


def preprocess_data(sentences):
    """
    数据预处理
    :process_text:
    """
    print("数据预处理...")
    process_text = pd.Series(sentences).apply(preprocess)
    print("数据预处理完毕！")
    return process_text


def w2v_model(sentences):
    """
    词向量模型Word2Vec
    ::
    """
    process_text = preprocess_data(sentences)
    w2v_input = process_text.tolist()
    w2v_input = [li for li in w2v_input if len(li) > 0]

    w2v = Word2Vec(w2v_input, size=300, min_count=1)  # 构建词向量模型
    print("开始训练模型……")
    w2v.train(w2v_input, total_examples=len(w2v_input), epochs=50)  # 训练词向量模型，(4234894, 4459000)
    print("词向量模型训练结束！")
    # print("单词总数量为：", len(w2v.wv.vocab))
    print("模型中单词数量为（unique）：", len(w2v.wv.vocab))  # 14076，w2v.wv.vocab单词对应词向量

    return w2v


def text_tokenizer(text, sentences):
    """
    tokenizer：向量化文本
    :return:
    """
    w2v = w2v_model(sentences)
    tokenizer = Tokenizer()  # 创建一个tokenizer对象
    tokenizer.fit_on_texts(text.tolist())
    vocab_size = len(tokenizer.word_index) + 1  # 词汇表长度，17781
    text_sequences = tokenizer.texts_to_sequences(text.tolist())  # 给单词编号
    print("text_sequences的长度为：", len(text_sequences))  # 222352
    max_length = 56  # 设置评论（单词）长度
    padding_text = pad_sequences(text_sequences, maxlen=max_length, padding='post')
    print("padding_text:", len(padding_text))

    # 出现在词汇表中但是没有出现在词向量模型中的单词张量设置为300个0填充
    embedding_index = {word: w2v.wv[word]
    if word in w2v.wv else np.zeros((300,))  # 此处300需要与前面设置的词向量表size一致
                       for word in tokenizer.word_index}
    embedding_matrix = np.zeros((vocab_size, 300))  # (17781, 300)
    for word in tokenizer.word_index:
        embedding_matrix[tokenizer.word_index.get(word)] = embedding_index.get(word)

    return vocab_size, padding_text, embedding_matrix


def split_data(padding_text):
    """
    将前面合并处理的数据集切分为：训练集、验证集、测试集
    """
    train_data, test_data = creat_dataset()
    # 特征划分
    train_padding = padding_text[:140000]
    val_padding = padding_text[140000:156060]
    test_padding = padding_text[156060:]
    # 标签划分
    train_labels = to_categorical(train_data.Sentiment.iloc[:140000])
    val_labels = to_categorical(train_data.Sentiment.iloc[140000:156060])
    print("训练集特征维度：", train_padding.shape)  # (140000, 56)
    return train_padding, val_padding, test_padding, train_labels, val_labels


def model(text, sentences):
    """
    keras模型
    """
    vocab_size, padding_text, embedding_matrix = text_tokenizer(text, sentences)
    train_padding, val_padding, test_padding, train_labels, val_labels = split_data(padding_text)
    model = Sequential()
    model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], embeddings_regularizer='l1', input_length=56,
                        trainable=True))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.save(r'D:\code\RottenTomatoes-Kaggle\RottenTomatoes-Kaggle\model\model.h5')
    # sgd = optimizers.SGD(lr=0.001)
    # model.load_weights(r'D:\code\RottenTomatoes-Kaggle\RottenTomatoes-Kaggle\model\model.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_padding, train_labels)
    history = model.fit(train_padding, train_labels, batch_size=300, validation_split=0.10, epochs=100, verbose=2)

    train_loss, train_accuracy = model.evaluate(train_padding, train_labels)
    print(f'训练集准确率：{train_accuracy * 100}%')

    val_loss, val_accuracy = model.evaluate(val_padding, val_labels)
    print(f'验证集准确率：{val_accuracy * 100}%')

    test_pre = model.predict_classes(test_padding)

    return test_pre


if __name__ == '__main__':

    text, sentences = process_dataset()  # 处理数据集
    pre = model(text, sentences)
    print("预测结果：\n", pre)
