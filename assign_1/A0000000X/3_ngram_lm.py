'''
    NUS CS4248 Assignment 1 - Objective 3 (n-gram Language Model)

    Class NgramLM for handling Objective 3
'''
import random, math

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
        self.n = n
        self.k = k
        self.word_count_dict = {}

    def update_corpus(self, text):
        ''' Updates the n-grams corpus based on text '''
        # TODO Write your code here
        pass

    def read_file(self, path):
        ''' Read the file and update the corpus  '''
        # TODO Write your code here
        pass

    def ngrams(self):
        ''' Returns ngrams of the text as list of pairs - [(sequence context, word)] '''
        # TODO Write your code here
        pass

    def add_padding(self):
        '''  Returns padded text '''
        # TODO Write your code here
        # Use '~' as your padding symbol
        pass

    def get_vocabulary(self):
        ''' Returns the vocabulary as set of words '''
        # TODO Write your code here
        pass

    def get_next_word_probability(self, text, word):
        ''' Returns the probability of word appearing after specified text '''
        # TODO Write your code here
        pass

    def generate_word(self, text):
        '''
        Returns a random word based on the specified text and n-grams learned
        by the model
        '''
        # TODO Write your code here
        pass

    def generate_text(self, length):
        ''' Returns text of the specified length based on the learned model '''
        # TODO Write your code here
        pass

    def perplexity(self, text):
        '''
        Returns the perplexity of text based on learned model

        Hint: To avoid numerical underflow, add logs instead of multiplying probabilities.
        Also handle the case when the LM assigns zero probabilities.
        '''
        # TODO Write your code here
        pass
