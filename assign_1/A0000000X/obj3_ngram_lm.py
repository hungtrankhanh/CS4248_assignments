'''
    NUS CS4248 Assignment 1 - Objective 3 (n-gram Language Model)

    Class NgramLM for handling Objective 3
'''
import random, math, re

class NgramLM(object):

    ADD_K_SMOOTHNG  = 0
    BACKOFF_SMOOTHING = 1
    INTERPOLATION_SMOOTHING = 2

    def __init__(self, n, k):
        '''
            Initialize your n-gram LM class

            Parameters:
                n (int) : order of the n-gram model
                k (float) : smoothing hyperparameter

        '''
        # Initialise other variables as necessary
        # TODO Write your code here
        self.text_corpus = []
        self.vocabulary = set()
        self.vocabulary_size = 0
        self.n = n
        self.k = k
        self.word_count_dict = {}
        self.pattern = r'\W+'
        self.lower_ngrams = []
        self.smoothing_mode = NgramLM.ADD_K_SMOOTHNG
        self.lambdas = []
        self.is_unigram = False

    def update_corpus(self, text):
        ''' Updates the n-grams corpus based on text '''
        # TODO Write your code here
        try :
            self.lower_ngrams.clear()
            self.vocabulary.clear()
            self.word_count_dict.clear()
            if self.n == 1:
                self.is_unigram = True
            else:
                self.is_unigram = False

            self.text_corpus = text
            self.ngrams()
            self.vocabulary = self.get_vocabulary()
            self.vocabulary_size = len(self.vocabulary)
            print(self.vocabulary)


        except Exception as e:
            print("Oops!", e.__class__, "occurred.")


    def read_file(self, path):
        ''' Read the file and update the corpus  '''
        # TODO Write your code here
        text_corpus = []
        with open(path, encoding='utf-8', errors='ignore') as f:
            line = f.readline()
            while line:
                norm_text = self.text_transformation(line.strip())
                text_corpus.append(norm_text)
                line = f.readline()
        self.update_corpus(text_corpus)

    def ngrams(self):
        ''' Returns ngrams of the text as list of pairs - [(sequence context, word)] '''
        # TODO Write your code here
        ngram_list = []
        padding_text_corpus = self.add_padding()
        if self.is_unigram:
            for text in padding_text_corpus:
                tokens, token_len = self.tokenize(text)
                for i in range(token_len):
                    word = tokens[i]
                    ngram_list.append(word)
                    if word in self.word_count_dict:
                        self.word_count_dict[word] += 1
                    else:
                        self.word_count_dict[word] = 1

            return ngram_list

        for text in padding_text_corpus:
            tokens, token_len = self.tokenize(text)
            n_len = token_len - (self.n - 1)
            for i in range(n_len):
                n_1_th_words = " ".join(tokens[i:i + (self.n - 1)])
                n_th_word = tokens[i + (self.n - 1)]
                item = (n_1_th_words, n_th_word)
                ngram_list.append(item)
                if n_1_th_words in self.word_count_dict:
                    nested_dict = self.word_count_dict[n_1_th_words]
                    if n_th_word in nested_dict:
                        nested_dict[n_th_word] += 1
                    else:
                        nested_dict[n_th_word] = 1
                else:
                    nested_dict = {n_th_word: 1}
                    self.word_count_dict[n_1_th_words] = nested_dict

        if self.smoothing_mode != NgramLM.ADD_K_SMOOTHNG:
            self.lower_order_ngrams(padding_text_corpus)

        return ngram_list

    def add_padding(self):
        '''  Returns padded text '''
        # TODO Write your code here
        # Use '~' as your padding symbol
        padding_text_corpus = []
        for text in self.text_corpus:
            padding_text = ("~ " * (self.n - 1)) + text + (" ~" * (self.n - 1))
            padding_text_corpus.append(padding_text)
        return padding_text_corpus

    def get_vocabulary(self):
        ''' Returns the vocabulary as set of words '''
        # TODO Write your code here
        vacabulary = set()
        for text in self.text_corpus:
            tokens, _ = self.tokenize(text)
            vacabulary.update(tokens)
        return vacabulary

    def get_next_word_probability(self, text, word):
        ''' Returns the probability of word appearing after specified text '''
        # TODO Write your code here
        tokens, n_len = self.tokenize(text)
        sequence_context = []
        sc_len = self.n - 1
        for i in range(sc_len):
            sequence_context.append("~")

        if n_len < sc_len:
            sequence_context[sc_len - n_len:sc_len] = tokens
        else:
            sequence_context[0: sc_len] = tokens[n_len-sc_len:n_len]
        next_word_prob = self.ngram_probability(sequence_context, word)
        return next_word_prob

        # return self.word_conditional_probability(sequence_context, word)

    def generate_word(self, text):
        '''
        Returns a random word based on the specified text and n-grams learned
        by the model
        '''
        # TODO Write your code here
        max_word_prob = 0.0
        possible_words = []
        norm_text = self.text_transformation(text)
        for w in self.vocabulary:
            candidate_text = norm_text + " " + w
            prob, _, _ = self.text_joint_probability(candidate_text)
            if prob > max_word_prob:
                possible_words.clear()
                possible_words.append(w)
                max_word_prob = prob
            elif prob == max_word_prob:
                possible_words.append(w)
        return random.choice(possible_words)

    def generate_text(self, length):
        ''' Returns text of the specified length based on the learned model '''
        # TODO Write your code here
        generated_text = random.choice(tuple(self.vocabulary))
        for i in range(length):
            word = self.generate_word(generated_text)
            generated_text = generated_text + " " + word

        return generated_text

    def perplexity(self, text):
        '''
        Returns the perplexity of text based on learned model

        Hint: To avoid numerical underflow, add logs instead of multiplying probabilities.
        Also handle the case when the LM assigns zero probabilities.
        '''
        # TODO Write your code here
        prob, _, n_len = self.text_joint_probability(text)
        return (1/prob)**(1/n_len)

    def ngram_probability(self, sequence_context, word):
        if self.smoothing_mode == NgramLM.ADD_K_SMOOTHNG:
            prob = self.add_k_smoothing_probability(sequence_context, word)
        elif self.smoothing_mode == NgramLM.BACKOFF_SMOOTHING:
            prob = self.backoff_smoothing_probability(sequence_context, word)
        elif self.smoothing_mode == NgramLM.INTERPOLATION_SMOOTHING:
            prob = self.interpolation_smoothing_probability(sequence_context, word)
        return prob

    def text_joint_probability(self, text):
        tokens, n_len = self.tokenize(text)
        sequence_context = []
        for i in range(self.n-1):
            sequence_context.append("~")
        sum_log_prob = 0.0
        for i in range(n_len):
            word = tokens[i]
            prob = self.ngram_probability(sequence_context, word)
            sum_log_prob += math.log(prob)
            if not self.is_unigram:
                sequence_context.pop(0)
                sequence_context.append(word)

        return math.exp(sum_log_prob), tokens, n_len

    def lower_order_ngrams(self, text_corpus):
        for ngram_th in range(1, self.n):
            ngram_dict = {}
            for text in text_corpus:
                tokens, token_len = self.tokenize(text)
                n_len = token_len - (ngram_th - 1)
                for i in range(n_len):
                    if ngram_th == 1:  # *unigram*
                        for i in range(n_len):
                            word = tokens[i]
                            if word in ngram_dict:
                                ngram_dict[word] += 1
                            else:
                                ngram_dict[word] = 1
                    else:
                        for i in range(n_len):
                            n_1_th_words = " ".join(tokens[i:i + (ngram_th - 1)])
                            n_th_word = tokens[i + (ngram_th - 1)]
                            if n_1_th_words in ngram_dict:
                                nested_dict = ngram_dict[n_1_th_words]
                                if n_th_word in nested_dict:
                                    nested_dict[n_th_word] += 1
                                else:
                                    nested_dict[n_th_word] = 1
                            else:
                                nested_dict = {n_th_word: 1}
                                ngram_dict[n_1_th_words] = nested_dict
            self.lower_ngrams.append(ngram_dict)


    def tokenize(self, text):
        split_text = re.split(self.pattern, text)
        tokens = [w for w in split_text if w]
        token_len = len(tokens)
        return tokens, token_len

    def set_smoothing_mode(self, mode, lambdas):
        if mode == "add_k":
            print("[obj3_ngram_lm] using add_k smoothing")
            self.smoothing_mode = NgramLM.ADD_K_SMOOTHNG
        elif mode == "backoff":
            print("[obj3_ngram_lm] using backoff smoothing")
            self.smoothing_mode = NgramLM.BACKOFF_SMOOTHING
        elif mode == "interpolation":
            self.lambdas = lambdas
            print("[obj3_ngram_lm] using interpolation smoothing:")
            self.smoothing_mode = NgramLM.INTERPOLATION_SMOOTHING
        else:
            print("[obj3_ngram_lm] using add_k smoothing")
            self.smoothing_mode = NgramLM.ADD_K_SMOOTHNG

        print("set_smoothing_mode = ", self.smoothing_mode)


    def add_k_smoothing_probability(self, sequence_context, word):

        if self.is_unigram:
            if word in self.word_count_dict:
                word_count = self.word_count_dict[word] + self.k
                all_word_count = sum(self.word_count_dict.values()) + self.k * self.vocabulary_size
            else:
                word_count = self.k
                all_word_count = sum(self.word_count_dict.values()) + self.k * (self.vocabulary_size + 1)

            prob = (word_count*1.0) / all_word_count
            return prob

        key = " ".join(sequence_context)
        if key in self.word_count_dict:
            nested_dict = self.word_count_dict[key]
            if word in nested_dict:
                context_word_count = nested_dict[word] + self.k
                context_count = sum(nested_dict.values()) + self.k*self.vocabulary_size
            else:
                context_word_count = self.k
                context_count = sum(nested_dict.values()) + self.k*(self.vocabulary_size + 1)

        else:
            context_word_count = self.k
            context_count = self.k*self.vocabulary_size
        prob = (context_word_count*1.0) / context_count
        return prob

    def backoff_smoothing_probability(self, sequence_context, word):
        if self.is_unigram:
            if word in self.word_count_dict:
                word_count = self.word_count_dict[word]
                all_word_count = sum(self.word_count_dict.values())
            else:
                word_count = 1
                all_word_count = sum(self.word_count_dict.values()) + 1

            prob = (word_count*1.0) / all_word_count
            return prob

        n_len = len(sequence_context)
        prob = 0.0
        for i in range(n_len+1):
            if i == 0:
                temp = sequence_context[i:n_len]
                key = " ".join(temp)
                if key in self.word_count_dict:
                    nested_dict = self.word_count_dict[key]
                    if word in nested_dict:
                        context_word_count = nested_dict[word]
                        context_count = sum(nested_dict.values())
                        prob = (context_word_count*1.0) / context_count
                        break
            elif i == n_len:
                unigram_dict = self.lower_ngrams[0]
                if word in unigram_dict:
                    prob = (unigram_dict[word] * 1.0) / sum(unigram_dict.values())
                else:
                    prob = 1.0/(sum(unigram_dict.values()) + 1)
            else:
                temp = sequence_context[i:n_len]
                key = " ".join(temp)
                ngram_dict = self.lower_ngrams[n_len-i]
                if key in ngram_dict:
                    nested_dict = ngram_dict[key]
                    if word in nested_dict:
                        context_word_count = nested_dict[word]
                        context_count = sum(nested_dict.values())
                        prob = (context_word_count*1.0) / context_count
                        break
        return prob

    def interpolation_smoothing_probability(self, sequence_context, word):
        if self.is_unigram:
            if word in self.word_count_dict:
                word_count = self.word_count_dict[word]
                all_word_count = sum(self.word_count_dict.values())
            else:
                word_count = 1
                all_word_count = sum(self.word_count_dict.values()) + 1

            prob = (word_count*1.0) / all_word_count
            return prob

        n_len = len(sequence_context)
        prob = 0.0
        for i in range(n_len + 1):
            temp_prob = 0.0
            if i == 0:
                temp = sequence_context[i:n_len]
                key = " ".join(temp)
                if key in self.word_count_dict:
                    nested_dict = self.word_count_dict[key]
                    if word in nested_dict:
                        context_word_count = nested_dict[word]
                        context_count = sum(nested_dict.values())
                        temp_prob = (context_word_count*1.0) / context_count
            elif i == n_len:
                unigram_dict = self.lower_ngrams[0]
                if word in unigram_dict:
                    temp_prob = (unigram_dict[word] * 1.0) / sum(unigram_dict.values())
                else:
                    temp_prob = 1.0 / (sum(unigram_dict.values()) + 1)
            else:
                temp = sequence_context[i:n_len]
                key = " ".join(temp)
                ngram_dict = self.lower_ngrams[n_len - i]
                if key in ngram_dict:
                    nested_dict = ngram_dict[key]
                    if word in nested_dict:
                        context_word_count = nested_dict[word]
                        context_count = sum(nested_dict.values())
                        temp_prob = (context_word_count*1.0) / context_count
            prob += self.lambdas[i]*temp_prob
        return prob

    def text_transformation(self, text):
        norm_text = text.casefold()
        norm_text = re.sub(r"(I|i)'m", r"\1 am", norm_text)
        norm_text = re.sub(r"(H|h)e's", r"\1e is", norm_text)
        norm_text = re.sub(r"(S|s)he's", r"\1he is", norm_text)
        norm_text = re.sub(r"(T|t)hat's", r"\1hat is", norm_text)
        norm_text = re.sub(r"(W|w)hat's", r"\1hat is", norm_text)
        norm_text = re.sub(r"(W|w)here's", r"\1here is", norm_text)
        norm_text = re.sub(r"(T|t)here's", r"\1here is", norm_text)
        norm_text = re.sub(r"\'ll", " will", norm_text)
        norm_text = re.sub(r"\'ve", " have", norm_text)
        norm_text = re.sub(r"\'re", " are", norm_text)
        norm_text = re.sub(r"\'d", " would", norm_text)
        norm_text = re.sub(r"won't", "will not", norm_text)
        norm_text = re.sub(r"can't", "can not", norm_text)
        return norm_text
