#!/usr/bin/env python3
from preprocess import *

def preprocess():
    train = read_file("_TR", "TRAIN", 1, 2500)
    test  = read_file("_TT", "TEST", 1, 1827)
    train_label = read_label(1, 2500)

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

    return (train_features, train_label, test_features)

train_features, train_label, test_features = preprocess()

import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

model1 = MultinomialNB()
model2 = LinearSVC()

model1.fit(train_features,train_label)
model2.fit(train_features,train_label)

result1 = model1.predict(test_features)
result2 = model2.predict(test_features)

with open('test-predict-NB', 'w') as data:
    data.write("Id,Prediction\n")
    for i, x in enumerate(result1):
        data.write("{},{}\n".format(i + 1, x))

with open('test-predict-SVC', 'w') as data:
    data.write("Id,Prediction\n")
    for i, x in enumerate(result2):
        data.write("{},{}\n".format(i + 1, x))
