'''
    NUS CS4248 Assignment 1 - Objective 1 (Tokenization)

    Class Tokenizer for handling Objective 1
'''
## Suggested libraries -- uncomment the below if you want to use these
## recommended resources and libraries.

from nltk.corpus import stopwords   # Requires NLTK in the include path.
import matplotlib.pyplot as plt     # Requires matplotlib to create plots.
import re
## List of stopwords
# import nltk
# nltk.download('stopwords')
STOPWORDS = stopwords.words('english') # type: list(str)
class Tokenizer:

    def __init__(self, path):
        with open(path, encoding='utf-8', errors='ignore') as f:
            self.text = f.read()

    def tokenize(self):
        ''' Returns a set of word tokens '''
        # TODO Modify the code here
        # words = re.split(r'[\s]+|\W[\W]+|[\W]\W+', self.text)
        words = re.split(r'[\s]+|\W[\W]+|[\W]\W+', self.text)
        words = [w for w in words if w]
        print("tokenize : ", words)
        return words

    def get_frequent_words(self, n):
        ''' Returns the most frequent unigrams from the text '''
        # TODO Modify the code here
        n_frequent_words_result = []

        try:
            words_count_dict = {}
            word_tokens = self.tokenize()
            for word in word_tokens:
                if word in words_count_dict:
                    words_count_dict[word] += 1
                else:
                    words_count_dict[word] = 1

            descending_ordered_words = []
            for word, count in words_count_dict.items():
                n_len = len(descending_ordered_words)
                if n_len == 0:
                    descending_ordered_words.append(word)
                else:
                    for idx in range(n_len):
                        temp_word = descending_ordered_words[idx]
                        temp_count = words_count_dict[temp_word]
                        if temp_count < count:
                            descending_ordered_words.insert(idx, word)
                            break
                        else:
                            if idx == (n_len-1):
                                descending_ordered_words.append(word)
            for idx in range(n):
                word = descending_ordered_words[idx]
                count = words_count_dict[word]
                item = (word, count)
                n_frequent_words_result.append(item)
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")


        return  n_frequent_words_result

    def plot_word_frequency(self):
        '''
        Plot relative frequency versus rank of word to check
        Zipf's law
        Relative frequency f = Number of times the word occurs /
                                Total number of word tokens
        Rank r = Index of the word according to word occurence list
        '''
        # TODO Modify the code here
        try:
            words_count_dict = {}
            word_tokens = self.tokenize()
            word_token_len = len(word_tokens)
            for word in word_tokens:
                if word in words_count_dict:
                    words_count_dict[word] += 1
                else:
                    words_count_dict[word] = 1

            descending_ordered_words = []
            for word, count in words_count_dict.items():
                n_len = len(descending_ordered_words)
                if n_len == 0:
                    descending_ordered_words.append(word)
                else:
                    for idx in range(n_len):
                        temp_word = descending_ordered_words[idx]
                        temp_count = words_count_dict[temp_word]
                        if temp_count < count:
                            descending_ordered_words.insert(idx, word)
                            break
                        else:
                            if idx == (n_len-1):
                                descending_ordered_words.append(word)

            word_frequency = []
            for word in descending_ordered_words:
                temp_count = words_count_dict[word]
                word_frequency.append(float(temp_count)/float(word_token_len))
            plt.plot(word_frequency)
            plt.xlabel('Word Rank')
            plt.ylabel('Word Frequency')
            plt.title("Zipf's law chart")
            plt.savefig('foo.png')
            plt.close()

        except Exception as e:
            print("Oops!", e.__class__, "occurred.")

    def remove_stopwords(self):
        ''' Removes stopwords from the text corpus '''
        # TODO Modify the code here
        print("STOPWORDS : ", STOPWORDS)
        for stopword in STOPWORDS:
            self.text = re.compile(r'(^|\s){}([\s]|$)'.format(stopword), re.IGNORECASE).sub(" ", self.text)
        print("remove_stopwords : ")

    def convert_lowercase(self):
        self.text = self.text.casefold()
