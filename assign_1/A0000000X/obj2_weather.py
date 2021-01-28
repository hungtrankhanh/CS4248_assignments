'''
    NUS CS4248 Assignment 1 - Objective 2 (Weather)

    Class Weather for handling Objective 2
'''
import re
class Weather:

    # Class Constants for responses.
    SINGAPORE_WEATHER = "Singapore: hot and humid."
    LONDON_WEATHER = "London: rainy and miserable."
    CAIRO_WEATHER = "Cairo: bone dry."
    DEFAULT = "Hmm. That’s nice."

    def __init__(self, path):
        self.text = path
        # with open(path, encoding='utf-8', errors='ignore') as f:
        #     self.text = f.read()

    def weather(self, text):
        '''
        Decide whether the input is about the weather and
        respond appropriately
        '''
        # TODO Modify the code here
        sg_re = "[Ss](ingapore|'pore|ing|[Gg])"
        ld_re = "[Ll]ondon"
        ca_re = "[Cc]airo"
        weather_re = "([Ww]hat|[Hh]ow)(.)*([Ww]eather)(.)*\?"
        if re.search(sg_re, text) and re.search(weather_re, text):
            return Weather.SINGAPORE_WEATHER
        if re.search(ld_re, text) and re.search(weather_re, text):
            return Weather.LONDON_WEATHER
        if re.search(ca_re, text) and re.search(weather_re, text):
            return Weather.CAIRO_WEATHER

        return Weather.DEFAULT
