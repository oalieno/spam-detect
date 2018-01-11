#!/usr/bin/env python3
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json

ps = PorterStemmer()

def read_file(dir_name, prefix_name, start, end):
    files = []
    for i in range(start, end + 1):
        with open("{}/{}_{}.eml".format(dir_name, prefix_name, i), "rb") as data:
            files.append(data.read())
    return files

def read_label(start, end):
    label = []
    with open('spam-mail.tr.label', 'r') as data:
        data.readline()
        for i in range(start, end + 1):
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

def percentage(value, total):
    return value / total * 100

def go(magic, train_features, train_label, test_features, test_label):
    model = magic
    model.fit(train_features,train_label)
    result = model.predict(test_features)
    conf = confusion_matrix(result, test_label)

    line = '-' * 13 + '+' + '-' * 12 + '+' + '-' * 13 + '+'

    print(line)
    print("             | actual:ham | actual:spam |")
    print(line)
    print("predict:ham  | {:10} | {:11} |".format(*conf[0]))
    print(line)
    print("predict:spam | {:10} | {:11} |".format(*conf[1]))
    print(line)
    print()

    print(line)
    print("             | actual:ham | actual:spam |")
    print(line)
    print("predict:ham  | {:9.6}% | {:10.6}% |".format(percentage(conf[0][0], conf.sum()), percentage(conf[0][1], conf.sum())))
    print(line)
    print("predict:spam | {:9.6}% | {:10.6}% |".format(percentage(conf[1][0], conf.sum()), percentage(conf[1][1], conf.sum())))
    print(line)
    print()

    print("accuracy: {:.6}%".format(percentage(conf.trace(), conf.sum())))
    print()

    return result

def store(filename, result):
    with open(filename, 'w') as data:
        data.write(json.dumps(result))

def get(filename):
    with open(filename, 'r') as data:
        ans = json.loads(data.read())
    return ans
