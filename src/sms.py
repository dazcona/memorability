import telebot
import os

telegram_token = '892268864:AAFmr9RcTbzZG_1IGlxeKtWTk5VsDVmM6ao'
telegram_chat_id = '624971481'

def send(message):
	bot = telebot.TeleBot(telegram_token)
	bot.config['api_key'] = telegram_token
	bot.send_message(int(telegram_chat_id), message)
	print('[INFO] Sending text...')


if __name__ == "__main__":
	send('Hi! This is a test!')
	print('Sending text...{}'.format(message))
