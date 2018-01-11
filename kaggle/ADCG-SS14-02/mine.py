#!/usr/bin/env python3
from preprocess import preprocess

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
