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

def preprocess():
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

    return (train_features, train_label, test_features)
