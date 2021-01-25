'''
    NUS CS4248 Assignment 1 - Objective 1 (Tokenization)

    Class Tokenizer for handling Objective 1
'''
## Suggested libraries -- uncomment the below if you want to use these
## recommended resources and libraries.

# from nltk.corpus import stopwords   # Requires NLTK in the include path.
# import matplotlib.pyplot as plt     # Requires matplotlib to create plots.

## List of stopwords
# STOPWORDS = stopwords.words('english') # type: list(str)
class Tokenizer:

    def __init__(self, path):
        with open(path, encoding='utf-8', errors='ignore') as f:
            self.text = f.read()

    def tokenize(self):
        ''' Returns a set of word tokens '''
        # TODO Modify the code here
        pass

    def get_frequent_words(self, n):
        ''' Returns the most frequent unigrams from the text '''
        # TODO Modify the code here
        pass

    def plot_word_frequency(self):
        '''
        Plot relative frequency versus rank of word to check
        Zipf's law
        Relative frequency f = Number of times the word occurs /
                                Total number of word tokens
        Rank r = Index of the word according to word occurence list
        '''
        # TODO Modify the code here
        pass

    def remove_stopwords(self):
        ''' Removes stopwords from the text corpus '''
        # TODO Modify the code here
        pass
