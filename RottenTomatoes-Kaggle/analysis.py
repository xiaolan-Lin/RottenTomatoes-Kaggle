import pandas as pd

pd.set_option('display.width', 10000)  # 设置显示的宽度为2000，防止输出内容被换行


def train_dataset():
    train_data = pd.read_csv(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\train.tsv', sep='\t')

    return train_data


def test_dataset():
    test_data = pd.read_csv(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\test.tsv', sep='\t')

    return test_data


if __name__ == '__main__':
    train_data = train_dataset()
    print(train_data.head())
    test_data = test_dataset()
    print(test_data.head())
    # print(len(train_data))
    # print(len(test_data))
    # print(train_data.SentenceId)
