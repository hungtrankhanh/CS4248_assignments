'''
    NUS CS4248 Assignment 1 - Objective 2 (Weather)

    Class Weather for handling Objective 2
'''
class Weather:

    # Class Constants for responses.
    SINGAPORE_WEATHER = "Singapore: hot and humid."
    LONDON_WEATHER = "London: rainy and miserable."
    CAIRO_WEATHER = "Cairo: bone dry."
    DEFAULT = "Hmm. That’s nice."

    def __init__(self, path):
        with open(path, encoding='utf-8', errors='ignore') as f:
            self.text = f.read()

    def weather(self,text):
        '''
        Decide whether the input is about the weather and
        respond appropriately
        '''
        # TODO Modify the code here
        pass
