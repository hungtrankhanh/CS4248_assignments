import numpy as np
import re
import pandas as pd

word_index_dict = {}
sentence_tokens_dict = {}
d_w_matrix = None
log_idf_w = None

short_word_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",
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

def tokenize(data_list):
    token_list = []
    for text in data_list:
        split_text = re.split(r'\W+', text)
        tokens = [w for w in split_text if len(w) > 1]
        token_list.append(tokens)
    return token_list

def normalization(data_list):
    new_data_list = []
    for text in data_list:
        norm_text = text.casefold()
        for key in short_word_dict:
            norm_text = re.sub(key, short_word_dict[key], norm_text)
        norm_text = re.sub(r'[\d]+[,.]?[\d]+', '', norm_text)
        norm_text = re.sub(r'(\W)([\w]+)s', '\1\2', norm_text)
        norm_text = re.sub(r'(\W)([\w]+)d', '\1\2', norm_text)

        new_data_list.append(norm_text)

    return new_data_list

def build_vocabulary(doc_token_list):
    vocabulary = set()
    word_index_dict = {}
    for tokens in doc_token_list:
        vocabulary.update(tokens)

    word_idx = 0
    for w in vocabulary:
        word_index_dict[w] = word_idx
        word_idx += 1
    V = len(vocabulary)
    print("vocabulary len :", V)
    return vocabulary, word_index_dict

def remove_stopwords(data_list, STOPWORDS):
    data_list_nostopwords = []
    for text in data_list:
        for stopword in STOPWORDS:
            text = re.compile(r'(^|\W+){}([\W+]|$)'.format(stopword), re.IGNORECASE).sub(" ", text)
        data_list_nostopwords.append(text)
    return data_list_nostopwords

def doc_word_matrix(doc_word_list, vocabulary, word_index_dict):
    N = len(doc_word_list)
    V = len(vocabulary)

    doc_word_count = np.zeros((N, V))
    doc_word_occur = np.zeros((N, V))
    for i in range(N):
        doc = doc_word_list[i]
        for w in doc:
            if w in vocabulary:
                idx = word_index_dict[w]
                doc_word_count[i][idx] += 1
                doc_word_occur[i][idx] = 1


    print("doc_word_count : ", doc_word_count.shape)
    return doc_word_count, doc_word_occur

def tf_df_matrix(doc_word_list, vocabulary, word_index_dict):
    doc_word_count, doc_word_occur = doc_word_matrix(doc_word_list, vocabulary, word_index_dict)
    N, V = doc_word_count.shape

    df_matrix = (N * 1.0) / np.sum(doc_word_occur, axis=0)
    log_df_matrix = np.log(df_matrix)

    log_tf_matrix = 1 + np.log(doc_word_count)
    log_tf_matrix[log_tf_matrix == -np.inf] = 0
    tf_df_matrix = log_tf_matrix / log_df_matrix
    print("tf_df_matrix : ", tf_df_matrix.shape)

    return tf_df_matrix


def top_4_grams_vocabulary(doc_word_list, label_list, top_n = [100, 50,50,50]):
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

    for doc in doc_word_list:
        w_len = len(doc)
        for i in range(w_len):
            unigram = doc[i]
            if unigram not in unigram_index_dict:
                unigram_index_dict[unigram] = u_idx
                unigram_list.append(unigram)
                u_idx += 1
            if i < w_len - 1:
                bigram = (doc[i], doc[i + 1])
                if bigram not in bigram_index_dict:
                    bigram_index_dict[bigram] = b_idx
                    bigram_list.append(bigram)
                    b_idx += 1
            if i < w_len - 2:
                trigram = (doc[i], doc[i + 1], doc[i + 2])
                if trigram not in trigram_index_dict:
                    trigram_index_dict[trigram] = t_idx
                    trigram_list.append(trigram)
                    t_idx += 1
            if i < w_len - 3:
                fourgram = (doc[i], doc[i + 1], doc[i + 2], doc[i + 3])
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
        label_idx = y + 1
        w_len = len(doc)
        for k in range(w_len):
            unigram = doc[k]
            if unigram in unigram_index_dict:
                u_idx = unigram_index_dict[unigram]
                unigram_count[label_idx][u_idx] += 1
            if k < w_len - 1:
                bigram = (doc[k], doc[k + 1])
                b_idx = bigram_index_dict[bigram]
                bigram_count[label_idx][b_idx] += 1
            if k < w_len - 2:
                trigram = (doc[k], doc[k + 1], doc[k + 2])
                t_idx = trigram_index_dict[trigram]
                trigram_count[label_idx][t_idx] += 1
            if k < w_len - 3:
                fourgram = (doc[k], doc[k + 1], doc[k + 2], doc[k + 3])
                f_idx = fourgram_index_dict[fourgram]
                fourgram_count[label_idx][f_idx] += 1

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

    return unigram_set, bigram_set, trigram_set, fourgram_set



def build_doc_word_matrix(self, X):
    N = len(X)
    V = len(self.vocabulary)
    self.d_w_matrix = np.zeros((N, V))
    self.log_idf_w = np.zeros((V, 1), dtype=int)
    d_w_related = np.zeros((N, V))

    for i in range(N):
        x = X[i]
        words, _ = self.tokenize(x)
        for w in words:
            if w in self.word_index_dict:
                idx = self.word_index_dict[w]
                self.d_w_matrix[i][idx] += 1
                d_w_related[i][idx] = 1

    self.log_idf_w = (N * 1.0) / np.sum(d_w_related, axis=0)
    self.log_idf_w = np.log(self.log_idf_w)

    print("self.log_idf_w : ", self.log_idf_w.shape)


def build_data_features(X, unigrams = None, bigrams = None, trigrams = None, four_grams = None):
    # vocabulary = [r"--",r"!",r"\?",r"\"",r";", r"[\d]+"]
    # present_tense = r"(is|are)"
    # past_tense = r"(was|were)"
    # perfect_tense = r"(have|has|had)[\s]+([a-z]+ed|been)"
    # future_tense = r"(will|would|shall|can|could)"
    # vocabulary.extend(unigrams)
    # vocabulary.extend(bigrams)
    # vocabulary.extend(trigrams)
    vocabulary = []
    if unigrams:
        unigrams_list = list(unigrams)
    if bigrams:
        bigrams_list  = list(bigrams)
    if trigrams:
        trigrams_list = list(trigrams)
    if four_grams:
        fourgrams_list = list(four_grams)


    N = len(X)
    V = len(unigrams) + len(bigrams) +len(trigrams) + len(four_grams)

    X_features = np.zeros((N, V), dtype=float)

    for n in range(N):
        x = X[n]
        n_x = len(x)
        w_idx = 0
        if unigrams_list:
            for v in unigrams_list:
                for w in x:
                    if v == w:
                        X_features[n][w_idx] += 1
                w_idx += 1

        if bigrams_list:
            for v in bigrams_list:
                for i in range(n_x):
                    if i < n_x - 1:
                        w = (x[i], x[i + 1])
                    if v == w:
                        X_features[n][w_idx] += 1
                w_idx += 1

        if trigrams_list:
            for v in trigrams_list:
                for i in range(n_x):
                    if i < n_x - 2:
                        w = (x[i], x[i + 1], x[i + 2])
                    if v == w:
                        X_features[n][w_idx] += 1
                w_idx += 1

        if fourgrams_list:
            for v in fourgrams_list:
                for i in range(n_x):
                    if i < n_x - 3:
                        w = (x[i], x[i + 1], x[i + 2], x[i + 3])
                    if v == w:
                        X_features[n][w_idx] += 1
                w_idx += 1

    print("self.train_word_vector : ", X_features.shape)
    return X_features

def train_word_vector(self):
    N, V = self.d_w_matrix.shape
    X_features = np.zeros((N, V), dtype=float)

    for n in range(N):
        x = self.d_w_matrix[n]
        for v in range(V):
            tf_w = x[v]
            if tf_w > 0:
                X_features[n][v] =  (1 + np.log(tf_w)) * self.log_idf_w[v]

    print("self.train_word_vector : ", X_features.shape)
    return  X_features

def test_word_vector(self, X):
    N = len(X)
    V = self.vocabulary
    X_features = np.zeros((N, V), dtype=float)

    d_w_matrix = np.zeros((N, V))

    for i in range(N):
        x = X[i]
        words, _ = self.tokenize(x)
        for w in words:
            if w in self.word_index_dict:
                idx = self.word_index_dict[w]
                d_w_matrix[i][idx] += 1

    for n in range(N):
        x = d_w_matrix[n]
        for v in range(V):
            tf_w = x[v]
            if tf_w > 0:
                X_features[n][v] =  (1 + np.log(tf_w)) * self.log_idf_w[v]

    print("self.test_word_vector : ", X_features.shape)
    return  X_features



def to_index(self, label):
    idx = 0
    if label == -1:
        idx = 0
    elif label == 0:
        idx = 1
    elif label == 1:
        idx = 2
    return idx

def to_label(self, idx):
    label = -1
    if idx == 0:
        label = -1
    elif idx == 1:
        label = 0
    elif idx == 2:
        label = 1
    return label