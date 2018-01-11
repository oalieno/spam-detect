#!/usr/bin/env python3
from utils import *

data = read_file("_TR", "TRAIN", 1, 2500)
data_label = read_label(1, 2500)

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from operator import itemgetter

result = get("project-pipe")

mark = []
for idx, val in enumerate(result):
    if val == 0: mark.append(idx)

data = itemgetter(*mark)(data)
data_label = itemgetter(*mark)(data_label)
train_features, train_label, test_features, test_label = preprocess(data, data_label, 0.3)

go(MultinomialNB(), train_features, train_label, test_features, test_label)
go(LinearSVC(), train_features, train_label, test_features, test_label)
