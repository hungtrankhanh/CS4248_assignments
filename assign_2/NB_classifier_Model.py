import numpy as np
import re
import pandas as pd
from data_reprocessing import *

# from nltk.corpus import stopwords   # Requires NLTK in the include path.
# import nltk
# nltk.download('stopwords')
# STOPWORDS = stopwords.words('english') # type: list(str)

class NB_classifier:
    def __init__(self, add_k = 1.0):
        self.n_label = 3
        self.add_k = add_k
        self.vocabulary = None
        self.word_index_dict = None
        self.doc_class_count = None
        self.word_class_count = None
        self.log_likelihood = None
        self.log_prior = None  # -1, 0, 1

    # def tokenize(self, text):
    #     split_text = re.split(self.pattern, text)
    #     tokens = [w for w in split_text if w]
    #     token_len = len(tokens)
    #     return tokens, token_len

    def train(self, X_train, y_train):
        X_train = tokenize(X_train)
        self.vocabulary, self.word_index_dict = build_vocabulary(X_train)

        N = len(y_train)
        V = len(self.vocabulary)

        self.doc_class_count = np.zeros((self.n_label, 1))
        self.word_class_count = np.zeros((self.n_label, V))
        self.log_likelihood = np.zeros((self.n_label, V))
        self.log_prior = np.zeros((self.n_label, 1))
        for i in range(N):
            x = X_train[i]
            y = y_train[i]
            self.doc_class_count[y] += 1
            for w in x:
                w_idx = self.word_index_dict[w]
                self.word_class_count[y][w_idx] += 1

        for k in range(self.n_label):
            prior_c = (self.doc_class_count[k] * 1.0) / N
            self.log_prior[k] = np.log(prior_c)
            total_count = np.sum(self.word_class_count[k]) + V*self.add_k
            for w in self.vocabulary:
                w_idx = self.word_index_dict[w]
                count_w_in_c = self.word_class_count[k][w_idx] + self.add_k
                prob = (count_w_in_c * 1.0) / total_count
                self.log_likelihood[k][w_idx] = np.log(prob)


    def predict(self, X_test):
        X_test = tokenize(X_test)
        N = len(X_test)
        results = []
        for j in range(N):
            x = X_test[j]
            log_probs = np.zeros((self.n_label, 1), dtype=float)
            for i in range(self.n_label):
                sum_c = self.log_prior[i]
                for w in x:
                    if w in self.vocabulary:
                        w_idx = self.word_index_dict[w]
                        sum_c += self.log_likelihood[i][w_idx]
                log_probs[i] = sum_c
            max_idx = np.argmax(log_probs)
            results.append(max_idx-1)
        print("predict_done")
        return results
