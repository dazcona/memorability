import telebot
import os

telegram_token = os.environ.get('TELEGRAM_TOKEN') # 892268864:AAFmr9RcTbzZG_1IGlxeKtWTk5VsDVmM6ao
telegram_chat_id = os.environ.get('TELEGRAM_ID') # 624971481

def send(message):
	bot = telebot.TeleBot(telegram_token)
	bot.config['api_key'] = telegram_token
	bot.send_message(int(telegram_chat_id), message)
	print('Sending text...{}'.format(message))