'''
    NUS CS4248 Assignment 1 - Objectives 1-3 Driver Code

    Classes Message, Echo, and helper functions for Slackbot
'''
from slack_sdk.rtm import RTMClient
import certifi
import ssl as ssl_lib
import re
from datetime import datetime
import json
# TODO add your imports here, so that you can have this main file use the
# from 1_tokenizer import *
from obj1_tokenizer import *
from obj2_weather import *
from obj3_ngram_lm import *

# classes defined for Objectives 1-3.

class Echo:
    def __init__(self, text):
        self.text = text

    def echo(text):
        ''' Echoes the text sent in '''
        reply = "You said: {}".format(text)
        return reply

class Message:

    def __init__(self, ts, username, text):
        self.ts = ts
        self.username = username
        self.text = text

    def toString(self):
        return f"Timestamp: {self.ts}, Username: {self.username}, Text: {self.text}"

    def toDict(self):
        data = {}
        data['Timestamp'] = self.ts
        data['Username'] = self.username
        data['Text'] = self.text
        return data

# List to keep track of messages sent and received
conversation = []

# TODO Make these whatever you want.
USERNAME = "CS4248_Bot_A0212253W"
USER_EMOJI = ":robot_face:"

# TODO Copy your Bot User OAuth-Access Token and paste it here
SLACK_TOKEN = "git "

def make_message(user_input):
    ''' Driver function - Parses the user_input, calls the appropriate classes and functions
    and returns the output to the make_message() function

    Example input: user_input = "OBJ0 echo_text=Hi there"
    '''
    # To stop the bot, simply enter the 'EXIT' command
    if user_input == 'EXIT':
        rtm_client.stop()
        with open('./conversation.json', 'w') as f:
            json.dump(conversation, f)
        return

    # Regex matching and calling appropriate classes and functions
    pattern_dict = {
        "OBJ0": r"OBJ0 echo_text=(?P<echo_text>.*)",
        "OBJ1": r"OBJ1 path=(?P<path>.*) n_top_words=(?P<n_top_words>\d+) lowercase=(?P<lowercase>YES|NO) stopwords=(?P<stopwords>YES|NO)",
        "OBJ2": r"OBJ2 (?P<input_text>.*)",
        "OBJ3": r"OBJ3 path=(?P<path>.*) smooth=(?P<smooth>.*) lambda=(?P<lambda>\[.*\]) n_gram=(?P<n_gram>\d) k=(?P<k>\d+(?:\.\d+)?) text=(?P<text>.*) next_word=(?P<next_word>.*) length=(?P<length>\d+)",
    }

    for key in pattern_dict.keys():
        match = re.match(pattern_dict[key], user_input)
        print("user_input :", user_input, " key:",key,"match:",match )
        if match:
            # Dictionary with key as argument name and value as argument value
            commands_dict = match.groupdict()
            if key == "OBJ0":
                print("[SUCCESS] Matched objective 0")
                echo = Echo()
                reply = echo.echo(commands_dict['echo_text'])
                break

            elif key == "OBJ1":
                print("[SUCCESS] Matched objective 1")
                filepath = commands_dict['path']
                n_top_words = commands_dict['n_top_words']
                is_lowercase = commands_dict['lowercase']
                is_stopwords = commands_dict['stopwords']
                obj1 = Tokenizer(filepath)
                if is_lowercase == 'YES':
                    obj1.convert_lowercase()
                if is_stopwords == 'YES':
                    obj1.remove_stopwords()

                n_top_word_list = obj1.get_frequent_words(int(n_top_words))
                obj1.plot_word_frequency()
                string_result = '\n'
                for item in n_top_word_list:
                    string_result += "('{}', {})\n".format(str(item[0]), str(item[1]))
                reply = string_result
                print("OBJ1 reply : ", reply)
                # TODO complete objective 1
                break

            elif key == "OBJ2":
                print("[SUCCESS] Matched objective 2")
                reply = Weather(commands_dict['input_text']).weather(commands_dict['input_text'])
                print("OJB2 reply : ", reply)
                # TODO complete objective 2
                break

            elif key == "OBJ3":
                print("[SUCCESS] Matched objective 3")
                # TODO complete objective 3
                filepath = commands_dict['path']
                smooth_type = commands_dict['smooth']
                lambda_list = commands_dict['lambda']
                n_gram = commands_dict['n_gram']
                add_k = commands_dict['k']
                text = commands_dict['text']
                next_word = commands_dict['next_word']
                length = commands_dict['length']

                ngramlm = NgramLM(int(n_gram), float(add_k))
                ngramlm.set_smoothing_mode(smooth_type, json.loads(lambda_list))
                ngramlm.read_file(filepath)
                generated_word = ngramlm.generate_word(text)
                print("Generated word: " + generated_word)
                next_word_prob = ngramlm.get_next_word_probability(text, next_word)
                print("Probability of next word: " + str(next_word_prob))
                perlexity_text = ngramlm.perplexity(text)
                print("Perplexity: " + str(perlexity_text))
                generated_text = ngramlm.generate_text(int(length))
                print("Generated text: " + generated_text)

                string_result = "\n"
                string_result += "Generated word: " + generated_word + "\n"
                string_result += "Probability of next word: " + str(next_word_prob) + "\n"
                string_result += "Perplexity: " + str(perlexity_text) + "\n"
                string_result += "Generated text: " + generated_text + "\n"
                reply = string_result
                break

            else:
                print("[ERROR] Did not match any commands!")
                reply = Echo.echo("[ERROR] Did not match any commands!")
        else:
            print("[ERROR] The command is wrong format!")
            reply = Echo.echo("[ERROR] The command is wrong format!")
    return reply


def do_respond(web_client, channel, text):
    # Post the message in Slack
    web_client.chat_postMessage(channel=channel,
                                username=USERNAME,
                                icon_emoji=USER_EMOJI,
                                text=make_message(text))

# ============== Message Events ============= #
# When a user sends a DM, the event type will be 'message'.
# Here we'll link the update_share callback to the 'message' event.
@RTMClient.run_on(event="message")

def message(**payload):
    """
    Call do_respond() with the appropriate information for all incoming
    direct messages to our bot.
    """
    web_client = payload["web_client"]

    # Getting information from the response
    data = payload["data"]
    channel_id = data.get("channel")
    text = data.get("text")
    subtype = data.get("subtype")
    ts = data['ts']
    user = data.get('username') if not data.get('user') else data.get('user')
    # Creating a Converstion object
    message = Message(ts, user, text)

    # Appending the converstion attributes to the logs
    conversation.append(message.toDict())

    if subtype == 'bot_message': return

    do_respond(web_client, channel_id, text)

# You probably won't need to modify any of the code below.
# It is used to appropriately install the bot.
def main():
    ssl_context = ssl_lib.create_default_context(cafile=certifi.where())
    # Real-time messaging client with Slack
    global rtm_client
    rtm_client = RTMClient(token=SLACK_TOKEN, ssl=ssl_context)
    try:
        print("[SUCCESS] Your bot is running!")
        rtm_client.start()
    except:
        print("[ERROR] Your bot is not running.")

if __name__ == "__main__":
    main()
