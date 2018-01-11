#!/usr/bin/env python3
from preprocess import *
from sklearn.model_selection import train_test_split

def preprocess(data, data_label, test_size):
    train, test, train_label, test_label = train_test_split(data, data_label, test_size = test_size, random_state = 0)

    train_tokens = list(map(tokenize, train))
    test_tokens  = list(map(tokenize, test))

    '''
    without stem : 124792
       with stem :  99114
    '''
    dictionary = build_dictionary(train_tokens + test_tokens)
    _dictionary = build_dictionary_inverse(dictionary)

    train_features = list(map(lambda x: tofeature(_dictionary, x), train_tokens))
    test_features  = list(map(lambda x: tofeature(_dictionary, x), test_tokens))

    return train_features, train_label, test_features, test_label

data = read_file("_TR", "TRAIN", 1, 2500)
data_label = read_label(1, 2500)
train_features, train_label, test_features, test_label = preprocess(data, data_label, 0.4)

import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

def percentage(value, total):
    return value / total * 100

def go(magic, train_features, train_label, test_features, test_label):
    model = magic
    model.fit(train_features,train_label)
    result = model.predict(test_features)
    conf = confusion_matrix(result, test_label)
    print('-' * 20)
    
    print("             actual:ham actual:spam")
    print("predict:ham  {:10}{:11}".format(*conf[0]))
    print("predict:spam {:10}{:11}".format(*conf[1]))
    
    print("             actual:ham actual:spam")
    print("predict:ham  {:9.6}%{:10.6}%".format(percentage(conf[0][0], conf.sum()), percentage(conf[0][1], conf.sum())))
    print("predict:spam {:9.6}%{:10.6}%".format(percentage(conf[1][0], conf.sum()), percentage(conf[1][1], conf.sum())))
    
    print("accuracy: {:.6}%".format(percentage(conf.trace(), conf.sum())))
    
    print('-' * 20)
    return result

result = go(MultinomialNB(), train_features, train_label, test_features, test_label)
go(LinearSVC(), train_features, train_label, test_features, test_label)

from operator import itemgetter

mark = []
for idx, val in enumerate(result):
    if val == 0: mark.append(idx)

data = itemgetter(*mark)(data)
data_label = itemgetter(*mark)(data_label)
train_features, train_label, test_features, test_label = preprocess(data, data_label, 0.3)

go(MultinomialNB(), train_features, train_label, test_features, test_label)
go(LinearSVC(), train_features, train_label, test_features, test_label)
