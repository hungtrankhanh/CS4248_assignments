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
        sing_re1 = "(what|how|tell|check|verify)(.)*(weather|climate|temperature|temp)(.)*([s](ingapore|'pore|ing|g))(.)*\??"
        sing_re2 = "([s](ingapore|'pore|ing|g))(.)*(what|how|tell|check|verify)(.)*(weather|climate|temperature|temp)(.)*\??"
        sing_re3 = "([s](ingapore|'pore|ing|g))(.)*(weather|climate|temperature|temp)(.)*(as|like|same)(.)*\?"
        sing_re4 = "(weather|climate|temperature|temp)(.)*([s](ingapore|'pore|ing|g))(.)*(as|like|same)(.)*\?"
        sing_re5 = "(weather|climate|temperature|temp)(.)*([s](ingapore|'pore|ing|g))(.)*\?"
        sing_re6 = "([s](ingapore|'pore|ing|g))(.)*(weather|climate|temperature|temp)(.)*\?"

        ld_re1 = "(what|how|tell|check|verify)(.)*(weather|climate|temperature|temp)(.)*(london)(.)*\??"
        ld_re2 = "(london)(.)*(what|how|tell|check|verify)(.)*(weather|climate|temperature|temp)(.)*\??"
        ld_re3 = "(london)(.)*(weather|climate|temperature|temp)(.)*(as|like|same)(.)*\?"
        ld_re4 = "(weather|climate|temperature|temp)(.)*(london)(.)*(as|like|same)(.)*\?"
        ld_re5 = "(weather|climate|temperature|temp)(.)*(london)(.)*\?"
        ld_re6 = "(london)(.)*(weather|climate|temperature|temp)(.)*\?"

        ca_re1 = "(what|how|tell|check|verify)(.)*(weather|climate|temperature|temp)(.)*(cairo)(.)*\??"
        ca_re2 = "(cairo)(.)*(what|how|tell|check|verify)(.)*(weather|climate|temperature|temp)(.)*\??"
        ca_re3 = "(cairo)(.)*(weather|climate|temperature|temp)(.)*(as|like|same)(.)*\?"
        ca_re4 = "(weather|climate|temperature|temp)(.)*(cairo)(.)*(as|like|same)(.)*\?"
        ca_re5 = "(weather|climate|temperature|temp)(.)*(cairo)(.)*\?"
        ca_re6 = "(cairo)(.)*(weather|climate|temperature|temp)(.)*\?"

        lower_case_text = text.casefold()
        if re.search(sing_re1, lower_case_text) \
                or re.search(sing_re2, lower_case_text) \
                or re.search(sing_re3, lower_case_text)\
                or re.search(sing_re4, lower_case_text)\
                or re.search(sing_re5, lower_case_text)\
                or re.search(sing_re6, lower_case_text):
            return Weather.SINGAPORE_WEATHER
        if re.search(ld_re1, lower_case_text)\
                or re.search(ld_re2, lower_case_text)\
                or re.search(ld_re3, lower_case_text)\
                or re.search(ld_re4, lower_case_text)\
                or re.search(ld_re5, lower_case_text)\
                or re.search(ld_re6, lower_case_text):
            return Weather.LONDON_WEATHER
        if re.search(ca_re1, lower_case_text)\
                or re.search(ca_re2, lower_case_text)\
                or re.search(ca_re3, lower_case_text)\
                or re.search(ca_re4, lower_case_text)\
                or re.search(ca_re5, lower_case_text)\
                or re.search(ca_re6, lower_case_text):
            return Weather.CAIRO_WEATHER

        return Weather.DEFAULT
