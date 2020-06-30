import pandas as pd


def train_dataset():
    train_data = pd.read_csv(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\train.tsv', sep='\t')

    return train_data


def test_dataset():
    test_data = pd.read_csv(r'D:\PycharmProjects\RottenTomatoes-Kaggle\data\test.tsv', sep='\t')

    return test_data


if __name__ == '__main__':
    train_data = train_dataset()
    test_data = test_dataset()
    print(len(train_data))
    print(len(test_data))
    # print(train_data.SentenceId)
