#!/usr/bin/env python3
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def read_file(dir_name, prefix_name, amounts):
    files = []
    for i in range(1, amounts + 1):
        with open("{}/{}_{}.eml".format(dir_name, prefix_name, i), "rb") as data:
            files.append(data.read())
    return files

def read_label():
    label = []
    with open('spam-mail.tr.label', 'r') as data:
        data.readline()
        for i in range(2500):
            line = data.readline()
            label.append(int(line.strip().split(',')[1].strip()))
    return label

def tokenize(text):
    tokens = word_tokenize(text.decode('ascii', 'ignore'))
    tokens = list(map(ps.stem, tokens))
    return tokens

def build_dictionary(tokens):
    dictionary = set([])
    for token in tokens:
        dictionary |= set(token)
    return list(dictionary)

def build_dictionary_inverse(dictionary):
    _dictionary = {}
    for i, d in enumerate(dictionary):
        _dictionary[d] = i
    return _dictionary

def tofeature(_dictionary, tokens):
    features = [0] * len(_dictionary)
    for token in tokens:
        features[_dictionary[token]] += 1
    return features

train = read_file("_TR", "TRAIN", 2500)
test  = read_file("_TT", "TEST", 1827)
train_label = read_label()

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

'''
# attempt to write to file and read it, but it is much slower O_O

with open('train-processed', 'w') as data:
    data.write('\n'.join(map(lambda x: ','.join(map(str, x)), train_features)))

with open('test-processed', 'w') as data:
    data.write('\n'.join(map(lambda x: ','.join(map(str, x)), test_features)))
'''

import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
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
