#!/usr/bin/env python3
from utils import *

data = read_file("_TR", "TRAIN", 1, 2500)
data_label = read_label(1, 2500)
train_features, train_label, test_features, test_label = preprocess(data, data_label, 0.4)

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

result = go(MultinomialNB(), train_features, train_label, test_features, test_label)
go(LinearSVC(), train_features, train_label, test_features, test_label)

store("project-pipe", result.tolist())
