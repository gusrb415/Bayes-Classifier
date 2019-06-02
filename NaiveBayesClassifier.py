"""
    Title: 나이브 베이지안 분류기
    References: https://medium.com/syncedreview/applying-multinomial-
                naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf
    Author: 김현규
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import sparse
from sklearn.metrics import accuracy_score


class NaiveBayesClassifier:
    def __init__(self, smoothing_factor=1.0):
        """
        나이브 베이지안 분류기
        가산 평활화를 위한 값을 받아서 평활화한다.
        평활화 설명: y가 1 일 때 'abc'란 단어가 1번 나타날때
        평활화 값이 0일 경우 Test단계에서 단어가 사전에 아예 없으면
        DivisionByZero에러가 뜨기때문에 0.001로 최소값을 준다.
        y가 0에선 그 단어가 안나타나면 그 값의 비중을 평활화 값만큼 준다
        (y=1, 단어='abc' 인 값은 평활화 값 + 1 값을 매긴다)
        :param smoothing_factor: Additive smoothing factor
        """
        self.k = smoothing_factor
        self.vocab_dict = None
        self.y_prob = None
        self.xy_prob = None

    def get_bag_of_words(self, x):
        """
        이미 만들어진 단어-아이디 형태의 Dictionary를 이용해서
        Input 단어의 갯수를 센다.
        :param x: 다수의 단어리스트를 포함한 2차원 리스트
        :return: 매 리스트별 존재하는 단어의 희소행렬
        """
        vocab_dict = self.vocab_dict
        if vocab_dict is None:
            raise Exception("Please Train the Model using 'fit' first")
        data_matrix = sparse.lil_matrix(arg1=(len(x), len(vocab_dict)))
        for i, doc in enumerate(x):
            for word in doc:
                word_id = vocab_dict.get(word, -1)
                if word_id != -1:
                    data_matrix[i, word_id] += 1

        # tocsr()은 Compressed Sparse Row를 뜻하며 더 빠른 계산을 위함이다
        return data_matrix.tocsr()

    def normalize(self, number_list):
        """
        Input 리스트에 Smoothing factor를 더하고
        전체 합이 1이 되도록 각각 나누어 준다.
        :param number_list: Numpy.array() 형태의 리스트
        :return: 정규화된 데이터
        """
        smoothing_factor = self.k
        if smoothing_factor <= 0:
            smoothing_factor = 1e-3
        length = number_list.shape[0]
        sum_list = np.sum(number_list, axis=0, keepdims=True)
        return (number_list + smoothing_factor) / (sum_list + smoothing_factor * length)

    def fit(self, x, y):
        """
        x에 있는 단어를 기록하고 id를 매긴다음
        모델을 트레이닝 시킨다
        :param x: 다수의 단어 리스트
        :param y: 각각의 x의 y값 (0부터 시작하는 정수)
        """
        vocab = set()
        for doc in x:
            for word in doc:
                vocab.add(word)

        vocab_dict = dict(zip(vocab, range(len(vocab))))
        self.vocab_dict = vocab_dict

        data_matrix = self.get_bag_of_words(x)
        data_delta = np.zeros(shape=(data_matrix.shape[0], max(y) + 1))
        for i, j in enumerate(y):
            data_delta[i, j] = 1

        self.y_prob = self.normalize(np.sum(data_delta, axis=0, keepdims=False))
        self.xy_prob = self.normalize(data_matrix.transpose().dot(data_delta))

    def predict(self, x):
        """
        x를 받아서 y_pred(레이블 별 확률) 값을 반환한다.
        :param x: 다수의 단어리스트
        :return: 각각의 x리스트의 레이블별 확률리스트
        """
        data_matrix = self.get_bag_of_words(x)
        log_y_prob = np.log(self.y_prob)
        log_xy_prob = np.log(self.xy_prob)
        log_dot_dxy = data_matrix.dot(log_xy_prob)
        log_predictions = np.expand_dims(log_y_prob, axis=0) + log_dot_dxy
        row_sum = np.sum(log_predictions, axis=1).repeat(log_predictions.shape[1]).reshape(log_predictions.shape)
        return 1 - log_predictions / row_sum

    def predict_classes(self, x):
        """
        x 값을 받아서 예상되는 레이블을 예측한다
        :param x: 다수의 단어리스트
        :return: 1차원 예상레이블 리스트
        """
        return np.argmax(self.predict(x), axis=1)

    def accuracy(self, x, y):
        """
        x, y 값을 받아서 레이블 예측 및 예측 정확도를 계산한다
        :param x: 다수의 단어리스트
        :param y: 단어리스트별 레이블값
        :return: 예측리스트, 정확도
        """
        predictions = self.predict_classes(x)
        return predictions, accuracy_score(predictions, y)


def load_data(path):
    """
    데이터 로드
    """
    train_doc_id, train_x, train_y = list(), list(), list()
    test_doc_id, test_x, test_y = list(), list(), list()
    label_list = list()
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.split('\t')
            curr_label = arr[1]
            if curr_label not in label_list:
                label_list.append(curr_label)
            if arr[0] == 'tr':
                train_y.append(label_list.index(curr_label))
                train_doc_id.append(arr[2])
                train_x.append(arr[3:])
            elif arr[0] == 'ts':
                test_y.append(label_list.index(curr_label))
                test_doc_id.append(arr[2])
                test_x.append(arr[3:])
            else:
                raise SyntaxError
    return train_doc_id, train_x, train_y, test_doc_id, test_x, test_y, label_list


def print_time_difference(timestamp=None):
    """
    시간 프린트 헬퍼 메소드
    """
    curr_time = datetime.now().timestamp()
    if timestamp is None:
        return curr_time
    print("Time taken: %.2f seconds" % float(curr_time - timestamp))
    print()
    return curr_time


def main():
    parser = argparse.ArgumentParser(description="Naive Bayes Classifier")
    parser.add_argument("--load_path", help="input the path of data file", default="blog_spam.txt")
    parser.add_argument("--save_path", help="input the path of output file", default="result.txt")
    args = parser.parse_args()

    time = print_time_difference()
    print("Loading Data")
    _, train_x, train_y, test_doc_id, test_x, test_y, label_list = load_data(args.load_path)
    time = print_time_difference(time)

    print("Training Model")
    model = NaiveBayesClassifier(smoothing_factor=1.0)
    model.fit(train_x, train_y)
    time = print_time_difference(time)

    print("Predicting Test Data")
    predictions, accuracy = model.accuracy(test_x, test_y)
    binary_predictions = [0 if i == label_list.index('정상') else 1 for i in predictions]
    binary_y_test = [0 if i == label_list.index('정상') else 1 for i in test_y]
    print("Model accuracy in Binary: %.2f%%" % (100 * accuracy_score(binary_predictions, binary_y_test)))
    print("Model accuracy in MultiClass: %.2f%%" % (100 * accuracy))
    time = print_time_difference(time)

    print("Writing Predictions to File (%s)" % args.save_path)
    df = pd.DataFrame()
    df['doc_id'] = test_doc_id
    df['label'] = ['정상' if i == 0 else '스팸' for i in binary_predictions]
    df.to_csv(args.save_path, sep='\t', index=False, header=False)
    print_time_difference(time)


if __name__ == '__main__':
    main()
