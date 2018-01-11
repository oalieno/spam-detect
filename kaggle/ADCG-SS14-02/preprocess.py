#!/usr/bin/env python3
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

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

