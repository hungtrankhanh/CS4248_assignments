import numpy as np
import re
import pandas as pd

word_index_dict = {}
sentence_tokens_dict = {}
d_w_matrix = None
log_idf_w = None

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
        # norm_text = re.sub(r'\b([\w]+)ies\b', '\1y', norm_text)
        # norm_text = re.sub(r'\b([\w]+)s\b', '\1', norm_text)
        # norm_text = re.sub(r'(\W)([\w]+)d', '\1\2', norm_text)
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

        unigram_set.update(["tag_exclamation","tag_question","tag_percent","tag_dot","tag_hyphen"])
        n_grams_vocabulary =  list(unigram_set)
        n_grams_vocabulary.extend(bigram_set)
        n_grams_vocabulary.extend(trigram_set)
        n_grams_vocabulary.extend(fourgram_set)
    else:
        n_grams_vocabulary =  unigram_list

    print(n_grams_vocabulary)
    print(len(n_grams_vocabulary))

    return n_grams_vocabulary, list(unigram_set)

def build_data_features(data_list, n_grams_vocabulary, w_dict = None, w_index_dict = None):
    N = len(data_list)
    V = len(n_grams_vocabulary)

    X_features = np.zeros((N, V), dtype=float)
    # X = tokenize(data_list)
    for n in range(N):
        # x = X[n]
        X = data_list[n]
        n_grams_X = n_grams_list(X)
        for n_gram in n_grams_X:
            if n_gram in w_index_dict:
                v = w_index_dict[n_gram]
                X_features[n][v] += 1
        # for v in range(V):
        #     w = n_grams_vocabulary[v]
        #     # if w in w_dict:
        #     #     grp = w_dict[w]
        #     #     for q in x:
        #     #         if q in grp:
        #     #             if q in w_index_dict:
        #     #                 q_idx = w_index_dict[q]
        #     #                 X_features[n][q_idx] += 1
        #     #             else:
        #     #                 X_features[n][v] += 1
        #     # else:
        #     #     regex = r"\b{}\b".format(w)
        #     #     X_features[n][v] += len(re.findall(regex, x2))
        #     # # for g in grp:
        #     regex = r"\b{}\b".format(w)
        #     X_features[n][v] += len(re.findall(regex, X))

    print("self.train_word_vector : ", X_features.shape)
    return X_features



