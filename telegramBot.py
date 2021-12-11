import requests
import json
import torch
import random
import logging

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

countries = []

def covidData(x):
    response = requests.get('https://api.covid19api.com/summary')
    if (response.status_code == 200):
        data = response.json()
        text = (
        '\nCountry: '+data['Countries'][x]['Country']+
        '\nTotal Confirmed: '+str(data['Countries'][x]['TotalConfirmed'])+
        "\nTotal Deaths: "+str(data["Countries"][x]["TotalDeaths"])+
        "\nTotal Recovered: "+str(data["Countries"][x]["TotalRecovered"])+
        "\nDate: "+data["Countries"][x]["Date"])
        return text

def covidGlobal():
    response = requests.get('https://api.covid19api.com/summary')
    if (response.status_code == 200):
        data = response.json()
        text = (
        '\nCountry: Global'+
        '\nTotal Confirmed: '+str(data['Global']['TotalConfirmed'])+
        "\nTotal Deaths: "+str(data["Global"]["TotalDeaths"])+
        "\nTotal Recovered: "+str(data["Global"]["TotalRecovered"])+
        "\nDate: "+data["Global"]["Date"])
        return text

def analyseUserResponse(userResponse):
    checkCountry = False
    checkGlobal = False
    tempCountry = []

    for word in userResponse:
        for country in countries:
            for words in country:
                if word==words:
                    tempCountry.append(word)
            tempCountry = list(dict.fromkeys(tempCountry))
            for country in countries:
                if tempCountry==country:
                    checkCountry=True
        if word=='global':
            checkGlobal = True
   
    if checkCountry==True:
        return covidData(countries.index(tempCountry))
    elif checkGlobal==True:
        return covidGlobal()
    else:
        # use nltk to chat

        spacing = " "
        userResponse = spacing.join(userResponse)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('COVID19-Chatbot/files/intentsTelegram.json', 'r') as json_data:
            intents = json.load(json_data)

        FILE = "COVID19-Chatbot/files/dataTelegram.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()

        while True:
            userResponse = tokenize(userResponse)
            X = bag_of_words(userResponse, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        return random.choice(intent["responses"])
            else:
                return "I don't understand..."

def startChat(update, context):
    """Echo the user message."""
    userResponse = update.message.text.lower()
    punctuations = ['.', '?', '!']
    
    for char in userResponse:
        for punctuation in punctuations:
            if char==punctuation:
                userResponse = userResponse.replace(char, '')

    userResponse = userResponse.split(" ")
    update.message.reply_text(analyseUserResponse(userResponse))

def info(update, context):
    text = (
        "Welcome to your chatbot assistant, my name is Bot. "
        "Feel free to ask me any data about COVID-19. "
        "\n\nMy creator is <b>Christensen Mario Frans</b>"
        "\nTelegram ID: <a href="">@rioontelegram</a>"
    )
    update.message.reply_text(text, parse_mode=ParseMode.HTML)

def filterCountryWords(countries):
    for country in countries:
        for word in country:
            if word=='the':
                country.remove(word)
    for country in countries:
        for word in country:
            if word=='and':
                country.remove(word)
    return countries

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


# driver code
print("Bot is starting...")

response = requests.get('https://api.covid19api.com/summary')
data = response.json()

for i in range(192): 
    country = data['Countries'][i]['Country'].lower()
    country = country.split(" ")

    if(country == ['brunei', 'darussalam']):
        country = ['brunei']
    elif(country == ['congo', '(brazzaville)']):
        country = ['congo', 'brazzaville']
    elif(country == ['congo', '(kinshasa)']):
        country = ['congo', 'kinshasa']
    elif(country == ['côte', "d'ivoire"]):
        country = ['cote', 'd', 'ivoire']
    elif(country == ['guinea-bissau']):
        country = ['guinea', 'bissau']
    elif(country == ['holy', 'see', '(vatican', 'city', 'state)']):
        country = ['vatican', 'city']
    elif(country == ['iran,', 'islamic', 'republic', 'of']):
        country = ['iran']
    elif(country == ['korea', '(south)']):
        country = ['south', 'korea']
    elif(country == ['lao', 'pdr']):
        country = ['laos']
    elif(country == ['macedonia,', 'republic', 'of']):
        country = ['macedonia']
    elif(country == ['micronesia,', 'federated', 'states', 'of']):
        country = ['micronesia']
    elif(country == ['palestinian', 'territory']):
        country =  ['palestine']
    elif(country == ['republic', 'of', 'kosovo']):
        country = ['kosovo']
    elif(country == ['russian', 'federation']):
        country = ['russia']
    elif(country == ['saint', 'vincent', 'and', 'grenadines']):
        country = ['saint', 'vincent', 'and', 'the', 'grenadines']
    elif(country == ['syrian', 'arab', 'republic', '(syria)']):
        country = ['syria']
    elif(country == ['taiwan,', 'republic', 'of', 'china']):
        country = ['taiwan']
    elif(country == ['tanzania,', 'united', 'republic', 'of']):
        country = ['tanzania']
    elif(country == ['timor-leste']):
        country = ['timor', 'leste']
    elif(country == ['venezuela', '(bolivarian', 'republic)']):
        country = ['venezuela']
    elif(country == ['viet', 'nam']):
        country = ['vietnam']

    countries.append(country)

countries = filterCountryWords(countries)

updater = Updater("1738842991:AAH5t3Xu5MmYJGlEIBTP5hpz0P4Vh5JkChw", use_context=True)

dp = updater.dispatcher
    
dp.add_handler(CommandHandler("info", info))
dp.add_handler(CommandHandler("start", info))
dp.add_handler(MessageHandler(Filters.text, startChat))

# log all errors
dp.add_error_handler(error)

# Start the Bot
updater.start_polling()
updater.idle()
