from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk
import spacy


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


nlp = spacy.load('en_core_web_md')  # 加载预训练模型
train_data = pd.read_csv(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\train.tsv', delimiter='\t')
test_data = pd.read_csv(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\test.tsv', delimiter='\t')
print("NLP以及数据集已加载…………")

text = pd.concat([train_data.Phrase, test_data.Phrase], axis=0)  # 合并训练集、测试集特征
combined = pd.concat([train_data, test_data], axis=0)  # 合并训练集、测试集

sentences = []
for i in combined.SentenceId.unique():
    sentences.append(combined.loc[combined.SentenceId == i]['Phrase'].iloc[0])


def preprocess(x):
    x = nlp(x.lower())  # 大小写
    stopwords = get_stop_words()
    tokens = [t for t in x]
    tokens = [t for t in tokens if t.text not in stopwords]
    tokens = [t.text for t in tokens if t.is_punct == False]
    tokens = [t for t in tokens if len(t) >= 3]
    tokens = [t.lemma_ for t in tokens]
    return tokens


print('数据预处理...')
process_text = pd.Series(sentences).apply(preprocess)

