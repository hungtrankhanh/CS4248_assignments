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
USERNAME = "CS4248 Bot of Awesomeness"
USER_EMOJI = ":dog:"

# TODO Copy your Bot User OAuth-Access Token and paste it here
SLACK_TOKEN = "xoxb-xxxxx-xxxxx-xxxxx"

# This is the function where you can make replies as a function
# of the message sent by the user
# You'll need to modify the code to call the functions that
# you've created in the rest of the exercises.
def make_message(text):
    # To stop the bot, simply enter the 'EXIT' command
    if text == 'EXIT':
        rtm_client.stop()
        with open('./conversation.json', 'w') as f:
            json.dump(conversation, f)
        return

    # TODO Write your code to route the messages to the appropriate class
    # depending on the first token.  You can start by trying to route a message
    # of the form "OBJ0 Hi there", to the Echo class above, and then delete
    # comment out the placeholder lines below.
    reply = "You said: {}".format(text)
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
