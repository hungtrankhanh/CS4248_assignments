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
SLACK_TOKEN = "xoxb-1655204110567-1670847146195-R8ve6x5qrw1lUVzvP23VYpsT"


# This is the function where you can make replies as a function
# of the message sent by the user
# You'll need to modify the code to call the functions that
# you've created in the rest of the exercises.
# def make_message(text):
#     # To stop the bot, simply enter the 'EXIT' command
#     if text == 'EXIT':
#         rtm_client.stop()
#         with open('./conversation.json', 'w') as f:
#             json.dump(conversation, f)
#         return
#
#     # TODO Write your code to route the messages to the appropriate class
#     print("text: ", text)
#     text = re.compile("\\s(\\s)+").sub(" ", text)
#     text = re.compile("[\\s]*=[\\s]*").sub("=", text)
#     print("text 2 : ", text)
#
#     arguments = text.split(" ")
#     print("arguments : ", arguments)
#     objective = arguments[0]
#     if objective == "OBJ1":
#         filepath = (arguments[1].split("=")[1]).strip()
#         is_lowercase = (arguments[3].split("=")[1]).strip()
#         is_stopwords = (arguments[4].split("=")[1]).strip()
#
#         print("command : objective :", objective, " filepath:", filepath, " lowercase:", is_lowercase, " stopwords:", is_stopwords)
#         obj1 = Tokenizer('textbooks/64378-0.txt')
#         if is_lowercase == 'YES':
#             obj1.convert_lowercase()
#
#         if is_stopwords == 'YES':
#             obj1.remove_stopwords()
#
#         n_frequent_words = obj1.get_frequent_words(10)
#         obj1.plot_word_frequency()
#     elif objective == "OBJ2":
#         print(objective)
#
#     string_result = ''
#     for item in n_frequent_words:
#         string_result += "('{}', {})\n".format(str(item[0]), str(item[1]))
#
#
#     # depending on the first token.  You can start by trying to route a message
#     # of the form "OBJ0 Hi there", to the Echo class above, and then delete
#     # comment out the placeholder lines below.
#     return Echo.echo(string_result)
def make_message(user_input):
    ''' Driver function - Parses the user_input, calls the appropriate classes and functions
    and returns the output to the make_message() function

    Example input: user_input = "OBJ0 echo_text=Hi there"
    '''
    print("-------------------------1")
    # To stop the bot, simply enter the 'EXIT' command
    if user_input == 'EXIT':
        rtm_client.stop()
        with open('./conversation.json', 'w') as f:
            json.dump(conversation, f)
        return
    print("-------------------------2")

    # Regex matching and calling appropriate classes and functions
    pattern_dict = {
        "OBJ0": r"OBJ0 echo_text=(?P<echo_text>.*)",
        "OBJ1": r"OBJ1 path=(?P<path>.*) n_top_words=(?P<n_top_words>\d+) lowercase=(?P<lowercase>YES|NO) stopwords=(?P<stopwords>YES|NO)",
        "OBJ2": r"OBJ2 (?P<input_text>.*)",
        "OBJ3": r"OBJ3 path=(?P<path>.*) smooth=(?P<smooth>.*) n_gram=(?P<n_gram>\d) k=(?P<k>\d+(?:\.\d+)?) text=(?P<text>.*)",
    }
    print("-------------------------3")

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
                print("----------1")
                obj1 = Tokenizer(filepath)
                print("----------2")

                if is_lowercase == 'YES':
                    obj1.convert_lowercase()
                if is_stopwords == 'YES':
                    obj1.remove_stopwords()
                print("----------3")

                n_top_word_list = obj1.get_frequent_words(int(n_top_words))
                obj1.plot_word_frequency()
                print("----------3 : n_top_word_list", n_top_word_list)

                string_result = ''
                for item in n_top_word_list:
                    string_result += "('{}', {})\n".format(str(item[0]), str(item[1]))
                print("return reyly: 0 000")
                reply = Echo.echo(string_result)
                print("return reyly: 0 ", reply)
                # TODO complete objective 1
                break

            elif key == "OBJ2":
                print("[SUCCESS] Matched objective 2")
                reply = Weather(commands_dict['input_text']).weather(commands_dict['input_text'])

                # TODO complete objective 2
                break

            elif key == "OBJ3":
                print("[SUCCESS] Matched objective 3")
                # TODO complete objective 3
                break

            else:
                print("[ERROR] Did not match any commands!")
    print("return reyly: ", reply)
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
    print("here 0")

    web_client = payload["web_client"]

    # Getting information from the response
    data = payload["data"]
    channel_id = data.get("channel")
    text = data.get("text")
    subtype = data.get("subtype")
    ts = data['ts']
    user = data.get('username') if not data.get('user') else data.get('user')
    print("here 1")
    # Creating a Converstion object
    message = Message(ts, user, text)
    print("here 2")

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
