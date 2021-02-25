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
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


np.random.seed(4248)

# TODO: Replace with your Student Number
_STUDENT_NUM = 'A0212253W'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

short_word_dict = {"ain't": "is not", "aren't": "are not","can't": "can not", "'cause": "because",
                    "could've": "could have", "couldn't": "could not", "didn't": "did not",
                    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                    "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is",
                    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                    "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is",
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                    "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would",
                    "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                    "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                    "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have",
                    "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                    "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                    "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                    "you're": "you are", "you've": "you have"}


'''Start Text Pre-processing'''
def tokenize(data_list):
    token_list = []
    for text in data_list:
        split_text = re.split(r'\W+', text)
        tokens = [w for w in split_text if len(w) > 0 and w != 's']
        token_list.append(tokens)
    return token_list

def normalization(data_list, STOPWORDS = None):
    new_data_list = []
    for text in data_list:
        norm_text = text.casefold()
        for key in short_word_dict:
            norm_text = re.sub(key, short_word_dict[key], norm_text)
        norm_text = re.sub(r'[\d]+[,.]?[\d]*', ' tag_number ', norm_text)
        norm_text = re.sub(r'!', ' tag_exclamation ', norm_text)
        norm_text = re.sub(r'\?', ' tag_question ', norm_text)
        norm_text = re.sub(r'%', ' tag_percent ', norm_text)
        norm_text = re.sub(r'\$', ' tag_dollar ', norm_text)
        norm_text = re.sub(r'(\.\.\.)', ' tag_dot ', norm_text)
        norm_text = re.sub(r'(--)', ' tag_hyphen ', norm_text)
        if STOPWORDS:
            norm_text = remove_stopwords(norm_text,STOPWORDS)
        word_list = re.split(r'\W+', norm_text)
        word_list = [w for w in word_list if len(w) > 0 and w != 's']

        norm_text = " ".join(word_list)
        new_data_list.append(norm_text)

    return new_data_list

def remove_stopwords(text, STOPWORDS):
    for stopword in STOPWORDS:
        text = re.compile(r'\b{}\b'.format(stopword), re.IGNORECASE).sub(" ", text)
    return text

def doc_word_matrix(data_list, vocabulary, w_index_dict):
    N = len(data_list)
    V = len(vocabulary)

    doc_word_count = np.zeros((N, V))
    doc_word_occur = np.zeros((N, V))
    for i in range(N):
        doc = n_grams_list(data_list[i])
        for w in doc:
            if w in w_index_dict:
                v = w_index_dict[w]
                doc_word_count[i][v] += 1
                doc_word_occur[i][v] = 1

    print("doc_word_count : ", doc_word_count.shape)
    return doc_word_count, doc_word_occur

def tf_df_matrix(data_list, vocabulary, w_index_dict):
    doc_word_count, doc_word_occur = doc_word_matrix(data_list, vocabulary, w_index_dict)
    N, V = doc_word_count.shape

    df_matrix = (N * 1.0) / np.sum(doc_word_occur, axis=0)
    log_df_matrix = np.log(df_matrix)

    log_tf_matrix = 1 + np.log(doc_word_count)
    log_tf_matrix[log_tf_matrix == -np.inf] = 0
    tf_df_matrix = log_tf_matrix / log_df_matrix
    print("tf_df_matrix : ", tf_df_matrix.shape)
    return tf_df_matrix

def n_grams_list(text):
    word_list = re.split(r'\W+', text)
    word_list = [w for w in word_list if w and w != 's']
    n_grams_list = []
    w_len = len(word_list)
    for i in range(w_len):
        n_grams_list.append(word_list[i])
        if i < w_len - 1:
            bigram = " ".join([word_list[i], word_list[i + 1]])
            n_grams_list.append(bigram)
        if i < w_len - 2:
            trigram = " ".join([word_list[i], word_list[i + 1], word_list[i + 2]])
            n_grams_list.append(trigram)
        if i < w_len - 3:
            fourgram = " ".join([word_list[i], word_list[i + 1], word_list[i + 2], word_list[i + 3]])
            n_grams_list.append(fourgram)

    return n_grams_list

def top_4_grams_vocabulary(doc_list, label_list, top_n = None):
    u_idx = 0
    b_idx = 0
    t_idx = 0
    f_idx = 0

    unigram_index_dict = {}
    unigram_list = []
    bigram_index_dict = {}
    bigram_list = []
    trigram_index_dict = {}
    trigram_list = []
    fourgram_index_dict = {}
    fourgram_list = []

    unigram_set = set()
    bigram_set = set()
    trigram_set = set()
    fourgram_set = set()

    doc_word_list = tokenize(doc_list)
    for doc in doc_word_list:
        w_len = len(doc)
        for i in range(w_len):
            unigram = doc[i]
            if unigram not in unigram_index_dict:
                unigram_index_dict[unigram] = u_idx
                unigram_list.append(unigram)
                u_idx += 1
            if i < w_len - 1:
                bigram = " ".join([doc[i], doc[i + 1]])
                if bigram not in bigram_index_dict:
                    bigram_index_dict[bigram] = b_idx
                    bigram_list.append(bigram)
                    b_idx += 1
            if i < w_len - 2:
                trigram = " ".join([doc[i], doc[i + 1], doc[i + 2]])
                if trigram not in trigram_index_dict:
                    trigram_index_dict[trigram] = t_idx
                    trigram_list.append(trigram)
                    t_idx += 1
            if i < w_len - 3:
                fourgram = " ".join([doc[i], doc[i + 1], doc[i + 2], doc[i + 3]])
                if fourgram not in fourgram_index_dict:
                    fourgram_index_dict[fourgram] = f_idx
                    fourgram_list.append(fourgram)
                    f_idx += 1

    V1 = len(unigram_list)
    V2 = len(bigram_list)
    V3 = len(trigram_list)
    V4 = len(fourgram_list)

    unigram_count = np.zeros((3, V1))
    bigram_count = np.zeros((3, V2))
    trigram_count = np.zeros((3, V3))
    fourgram_count = np.zeros((3, V4))



    for doc, y in zip(doc_word_list, label_list):
        label_idx = y
        w_len = len(doc)
        for k in range(w_len):
            unigram = doc[k]
            if unigram in unigram_index_dict:
                u_idx = unigram_index_dict[unigram]
                unigram_count[label_idx][u_idx] += 1
            if k < w_len - 1:
                bigram = " ".join([doc[k], doc[k + 1]])
                b_idx = bigram_index_dict[bigram]
                bigram_count[label_idx][b_idx] += 1
            if k < w_len - 2:
                trigram = " ".join([doc[k], doc[k + 1], doc[k + 2]])
                t_idx = trigram_index_dict[trigram]
                trigram_count[label_idx][t_idx] += 1
            if k < w_len - 3:
                fourgram = " ".join([doc[k], doc[k + 1], doc[k + 2], doc[k + 3]])
                f_idx = fourgram_index_dict[fourgram]
                fourgram_count[label_idx][f_idx] += 1

    if top_n:
        for i in range(3):
            u_class = unigram_count[i]
            b_class = bigram_count[i]
            t_class = trigram_count[i]
            f_class = fourgram_count[i]

            u_indices = (-u_class).argsort()[:top_n[0]]
            for u_i in u_indices:
                unigram_set.add(unigram_list[u_i])

            b_indices = (-b_class).argsort()[:top_n[1]]
            for b_i in b_indices:
                bigram_set.add(bigram_list[b_i])

            t_indices = (-t_class).argsort()[:top_n[2]]
            for t_i in t_indices:
                trigram_set.add(trigram_list[t_i])

            f_indices = (-f_class).argsort()[:top_n[3]]
            for f_i in f_indices:
                fourgram_set.add(fourgram_list[f_i])

        unigram_set.update(["tag_number", "tag_exclamation", "tag_question", "tag_percent", "tag_dollar", "tag_dot", "tag_hyphen"])
        n_grams_vocabulary = list(unigram_set)
        n_grams_vocabulary.extend(bigram_set)
        n_grams_vocabulary.extend(trigram_set)
        n_grams_vocabulary.extend(fourgram_set)
    else:
        n_grams_vocabulary = list(unigram_list)

    print(n_grams_vocabulary)
    print(len(n_grams_vocabulary))

    return n_grams_vocabulary, list(unigram_set)

def build_data_features(data_list, n_grams_vocabulary, w_dict = None, w_index_dict = None):
    N = len(data_list)
    V = len(n_grams_vocabulary)

    X_features = np.zeros((N, V), dtype=float)
    for n in range(N):
        X = data_list[n]
        n_grams_X = n_grams_list(X)
        for n_gram in n_grams_X:
            if n_gram in w_index_dict:
                v = w_index_dict[n_gram]
                X_features[n][v] += 1

    return X_features

'''End Text Pre-processing'''


'''Start Neural Network Model'''
class TextClassifier(nn.Module):
  def __init__(self, D_in, H1, H2, D_out):
      super().__init__()
      self.linear1 = nn.Linear(D_in, H1)
      self.weighs_init(self.linear1)

      self.linear2 = nn.Linear(H1, H2)
      self.weighs_init(self.linear2)

      self.linear3 = nn.Linear(H2, D_out)
      self.weighs_init(self.linear3)

      self.criterion = None
      self.optimizer = None
      self.batch_size = 32
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.n_grams_vocabulary = None
      self.group_word_dict = {}
      self.word_index_dict = {}

  def weighs_init(self,m):
      n = m.in_features
      y = 1.0/np.sqrt(n)
      m.weight.data.uniform_(-y , y)
      m.bias.data.fill_(0.0)

  def forward(self, x):
      x = F.relu(self.linear1(x))
      x = F.relu(self.linear2(x))
      x = self.linear3(x)
      return x

  def loss_function(self, lr = 0.0001, batch_size = 64):
      self.batch_size = batch_size
      self.criterion = nn.CrossEntropyLoss()
      self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
      self.to(self.device)


  def predict(self, X_test):
      X_features = build_data_features(X_test, self.n_grams_vocabulary, None,  self.word_index_dict)
      pred_result = []
      for x in X_features:
          x = torch.tensor(x, dtype=torch.float32).to(self.device)
          output = self(x)
          _, pred = torch.max(output, 0)
          pred_result.append(pred.item() - 1)

      return pred_result

  def train(self, X_data, y_data):
      self.n_grams_vocabulary, unigram_list = top_4_grams_vocabulary(X_data, y_data, [1500, 1500, 1000, 500])
      word_idx = 0
      for w in self.n_grams_vocabulary:
          self.word_index_dict[w] = word_idx
          word_idx += 1

      X_features = build_data_features(X_data, self.n_grams_vocabulary, None,  self.word_index_dict)

      kf = KFold(n_splits=8, shuffle=True, random_state=42)
      kfold_num = 1
      for train_index, val_index in kf.split(y_data):

          X_train = X_features[train_index]
          y_train = y_data[train_index]
          X_val = X_features[val_index]
          y_val = y_data[val_index]

          X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
          y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
          X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
          y_val = torch.tensor(y_val, dtype=torch.long).to(self.device)

          print("model train==========Fold number = ",  kfold_num, " - X_train shape = ", X_train.shape)

          data_train = torch.utils.data.TensorDataset(X_train, y_train)
          data_val = torch.utils.data.TensorDataset(X_val, y_val)
          train_loader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
          valid_loader = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, shuffle=False)

          ##################################################

          if kfold_num > 1:
              epochs = 20
          else:
              epochs = 40

          running_loss_history = []
          running_corrects_history = []
          val_running_loss_history = []
          val_running_corrects_history = []
          for e in range(epochs):
              running_loss = 0.0
              running_corrects = 0.0
              val_running_loss = 0.0
              val_running_corrects = 0.0
              training_loader_len = 0
              validation_loader_len = 0
              for i, (inputs, labels) in enumerate(train_loader):
                  training_loader_len += len(labels)
                  outputs = self(inputs)
                  loss = self.criterion(outputs, labels)
                  l1 = 0
                  for p in self.parameters():
                      l1 = l1 + p.abs().sum()
                  loss = loss + 0.0001 * l1
                  self.optimizer.zero_grad()
                  loss.backward()
                  self.optimizer.step()

                  _, preds = torch.max(outputs, 1)
                  running_loss += loss.item()
                  running_corrects += torch.sum(preds == labels.data)
              else:
                  with torch.no_grad():
                      for i ,(val_inputs, val_labels) in enumerate(valid_loader):
                          validation_loader_len += len(val_labels)
                          val_outputs = self(val_inputs)
                          val_loss = self.criterion(val_outputs, val_labels)

                          _, val_preds = torch.max(val_outputs, 1)
                          val_running_loss += val_loss.item()
                          val_running_corrects += torch.sum(val_preds == val_labels.data)

                  epoch_loss = running_loss / training_loader_len
                  epoch_acc = running_corrects / training_loader_len
                  running_loss_history.append(epoch_loss)
                  running_corrects_history.append(epoch_acc)

                  val_epoch_loss = val_running_loss / validation_loader_len
                  val_epoch_acc = val_running_corrects / validation_loader_len
                  val_running_loss_history.append(val_epoch_loss)
                  val_running_corrects_history.append(val_epoch_acc)
                  print('epoch : ', e)
                  print('training_loss: {:.4f}, {:.4f} '.format(epoch_loss, epoch_acc))
                  print('validation_loss: {:.4f}, {:.4f} '.format(val_epoch_loss, val_epoch_acc))

          ##################################################

          kfold_num += 1

'''End Neural Network Model'''


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
    train = pd.read_csv('train.csv')

    #data train
    X_train = normalization(train['Text'].tolist())
    y_train = train['Verdict'].to_numpy()
    y_train = y_train + 1

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
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")
    print("program end !")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
