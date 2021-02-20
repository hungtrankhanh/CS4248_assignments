#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import pandas as pd
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import numpy as np

from nltk.corpus import stopwords   # Requires NLTK in the include path.
## List of stopwords
import nltk
# nltk.download('stopwords')
STOPWORDS = stopwords.words('english') # type: list(str)

from sklearn.metrics import f1_score

from NB_classifier_Model import NB_classifier
from data_reprocessing import *
from nn_model import *
from data_loader import *


# TODO: Replace with your Student Number
_STUDENT_NUM = 'A0212253W'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    model.train(X_train, y_train)


def predict(model, X_test):
    ''' TODO: make your prediction here '''
    return model.predict(X_test)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train_mode = 2
    train = pd.read_csv('train.csv')

    #data train
    X_train_data = normalization(train['Text'].tolist())
    # X_train_data = remove_stopwords(X_train_data, STOPWORDS)
    X_train = tokenize(X_train_data)
    y_train = train['Verdict'].to_numpy()

    vocabulary, word_index_dict = build_vocabulary(X_train)
    print("---vocabulary = ", vocabulary)

    model = None
    if train_mode == 1:
        model = NB_classifier(vocabulary, word_index_dict, 3.0)
    else :
        # tf_df_matrix(X_train, vocabulary, word_index_dict)
        model = TextClassifier(434,200, 50, 3)
        model.loss_function(lr=0.0001, batch_size=256)


    train_model(model, X_train, y_train)
    # test your model'
    y_pred = predict(model, X_train)
    print("---y_pred = ", y_pred)

    print("-----------------3333")

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('test.csv')
    X_test_data = normalization(test['Text'].tolist())
    X_test = tokenize(X_test_data)
    y_pred = predict(model, X_test)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
