'''
    NUS CS4248 Assignment 1 - Objective 3 (n-gram Language Model)

    Class NgramLM for handling Objective 3
'''
import random, math, re

class NgramLM(object):

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
        self.pattern = r"\s+|[.,!?]"

    def update_corpus(self, text):
        ''' Updates the n-grams corpus based on text '''
        # TODO Write your code here
        self.text_corpus = text
        self.ngrams()
        self.vocabulary = self.get_vocabulary()
        self.vocabulary_size = len(self.vocabulary)

    def read_file(self, path):
        ''' Read the file and update the corpus  '''
        # TODO Write your code here
        text_corpus = []
        with open(path, encoding='utf-8', errors='ignore') as f:
            line = f.readline()
            while line:
                text_corpus.append(line.strip())
                line = f.readline()
        self.update_corpus(text_corpus)

    def ngrams(self):
        ''' Returns ngrams of the text as list of pairs - [(sequence context, word)] '''
        # TODO Write your code here
        ngram_list = []
        padding_text_corpus = self.add_padding()
        for text in padding_text_corpus:
            tokens, token_len = self.tokenize(text)
            n_len = token_len - self.n
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
        print(ngram_list)
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
        for i in range(self.n-1):
            sequence_context.append("~")
        sum_log_prob = 0.0

        for i in range(n_len):
            word_text = tokens[i]
            n_1_gram_context = " ".join(sequence_context)
            prob = self.word_conditional_probability(n_1_gram_context, word_text)
            sum_log_prob += math.log(prob)
            sequence_context.pop(0)
            sequence_context.append(word_text)

        word_previous_context = " ".join(sequence_context)
        word_prob = self.word_conditional_probability(word_previous_context, word)
        sum_log_prob += math.log(word_prob)
        return math.exp(sum_log_prob)

        # return self.word_conditional_probability(sequence_context, word)

    def generate_word(self, text):
        '''
        Returns a random word based on the specified text and n-grams learned
        by the model
        '''
        # TODO Write your code here
        max_word_prob = 0.0
        possible_words = []
        for w in self.vocabulary:
            prob = self.get_next_word_probability(text, w)
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
        tokens, n_len = self.tokenize(text)
        sequence_context = []
        for i in range(self.n-1):
            sequence_context.append("~")
        sum_log_prob = 0.0

        for i in range(n_len):
            word = tokens[i]
            n_1_gram_context = " ".join(sequence_context)
            prob = self.word_conditional_probability(n_1_gram_context, word)
            sum_log_prob += math.log(prob)
            sequence_context.pop(0)
            sequence_context.append(word)

        perlexity_text = (1/math.exp(sum_log_prob))**(1/n_len)
        return perlexity_text

    def word_conditional_probability(self, sequence_context, word):
        if sequence_context in self.word_count_dict:
            nested_dict = self.word_count_dict[sequence_context]
            n_gram_count = self.k
            if word in nested_dict:
                n_gram_count += nested_dict[word]
            n_1_gram_count = sum(nested_dict.values()) + self.k*self.vocabulary_size
        else:
            n_gram_count = self.k
            n_1_gram_count = self.k*self.vocabulary_size
        prob = n_gram_count / n_1_gram_count

        return prob

    def tokenize(self, text):
        split_text = re.split(self.pattern, text)
        tokens = [w for w in split_text if w]
        token_len = len(tokens)
        return tokens, token_len

