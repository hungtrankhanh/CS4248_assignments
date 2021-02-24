import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time
from gensim.models import Word2Vec


from data_reprocessing import *
np.random.seed(4248)


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
      # self.optimizer = torch.optim.Adam(self.parameters(), lr)
      self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
      # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
      self.to(self.device)


  def predict(self, X_test):
      # w_tf_df_matrix = tf_df_matrix(X_test, self.n_grams_vocabulary, self.word_index_dict)
      X_features = build_data_features(X_test, self.n_grams_vocabulary, None,  self.word_index_dict)
      # X_features = X_features*w_tf_df_matrix
      pred_result = []
      for x in X_features:
          x = torch.tensor(x, dtype=torch.float32).to(self.device)
          output = self(x)
          _, pred = torch.max(output, 0)
          pred_result.append(pred.item() - 1)

      return pred_result

  def train(self, X_data, y_data):
      # sentences = tokenize(X_data)
      # word2vec_model = Word2Vec(sentences=sentences, min_count =2, sg=1, size=100, window = 8, workers=4)

      self.n_grams_vocabulary, unigram_list = top_4_grams_vocabulary(X_data, y_data, [1500, 1500,1000,500])#, top_n = [160, 50,50,50]

      word_idx = 0
      for w in self.n_grams_vocabulary:
          self.word_index_dict[w] = word_idx
          word_idx += 1

      # for w in unigram_list:
      #     similar_list = word2vec_model.wv.most_similar(w)
      #     similar_w = []
      #     for item in similar_list:
      #         similar_w.append(item[0])
      #     similar_w.append(w)
      #     self.group_word_dict[w] = similar_w
      #
      # print("w_dict : ", len(self.n_grams_vocabulary))
      # print("w_dict : ", len(self.group_word_dict))


      # w_tf_df_matrix = tf_df_matrix(X_data, self.n_grams_vocabulary, self.word_index_dict)

      X_features = build_data_features(X_data, self.n_grams_vocabulary, None,  self.word_index_dict)
      # X_features = X_features*w_tf_df_matrix

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

          print("==============================Fold number = ",  kfold_num, " - ", X_train.shape)

          data_train = torch.utils.data.TensorDataset(X_train, y_train)
          data_val = torch.utils.data.TensorDataset(X_val, y_val)
          train_loader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
          valid_loader = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, shuffle=False)

          ##################################################

          # #for LeNet
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
              # train_len = len(data_train)
              # train_loop = int(np.floor(train_len/128)) + 1
              # print("train_len :", train_len ,"train_loop:", train_loop)
              training_loader_len = 0
              validation_loader_len = 0
              for i, (inputs, labels) in enumerate(train_loader):
                  training_loader_len += len(labels)
                  outputs = self(inputs)
                  loss = self.criterion(outputs, labels)
                  # optimizer.zero_grad()
                  # loss.backward()
                  # optimizer.step()

                  #  loss = mse(pred, target)
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
                      # val_len = len(data_val)
                      # val_loop = int(np.floor(val_len/128)) + 1
                      for i ,(val_inputs, val_labels) in enumerate(valid_loader):
                          validation_loader_len += len(val_labels)
                          # for i in range(val_loop):
                          # val_inputs, val_labels = data_val.sample()
                          val_outputs = self(val_inputs)
                          val_loss = self.criterion(val_outputs, val_labels)

                          _, val_preds = torch.max(val_outputs, 1)
                          val_running_loss += val_loss.item()
                          val_running_corrects += torch.sum(val_preds == val_labels.data)

                  # training_loader_len = len(training_loader)
                  print("training_loader_len : ", training_loader_len)
                  epoch_loss = running_loss / training_loader_len
                  epoch_acc = running_corrects / training_loader_len
                  running_loss_history.append(epoch_loss)
                  running_corrects_history.append(epoch_acc)

                  # validation_loader_len = len(val_loader)
                  val_epoch_loss = val_running_loss / validation_loader_len
                  val_epoch_acc = val_running_corrects / validation_loader_len
                  val_running_loss_history.append(val_epoch_loss)
                  val_running_corrects_history.append(val_epoch_acc)
                  print('epoch : ', e)
                  print('training_loss: {:.4f}, {:.4f} '.format(epoch_loss, epoch_acc))
                  print('validation_loss: {:.4f}, {:.4f} '.format(val_epoch_loss, val_epoch_acc))

          ##################################################

          kfold_num += 1