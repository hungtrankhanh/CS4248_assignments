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
    train_mode = 1 #0 : BN model, 1: NN model
    train = pd.read_csv('train.csv')

    #data train
    X_train = normalization(train['Text'].tolist())
    y_train = train['Verdict'].to_numpy()
    y_train = y_train + 1

    model = None
    if train_mode == 1:
        model = NB_classifier(3.0)
    else :
        model = TextClassifier(8189, 150, 30, 3)
        model.loss_function(lr=0.001, batch_size=150)


    train_model(model, X_train, y_train)
    # test your model'
    y_pred = predict(model, X_train)


    # Use f1-macro as the metric
    y_train = y_train - 1
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('test.csv')
    X_test = normalization(test['Text'].tolist())
    y_pred = predict(model, X_test)
    print("---y_pred = ", y_pred)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
