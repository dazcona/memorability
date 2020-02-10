import telebot
import os

telegram_token = 'Token here!'
telegram_chat_id = 'Chat id here!'

def send(message):
	bot = telebot.TeleBot(telegram_token)
	bot.config['api_key'] = telegram_token
	bot.send_message(int(telegram_chat_id), message)
	print('[INFO] Sending text...')


if __name__ == "__main__":
	send('Hi! This is a test!')
	print('Sending text...{}'.format(message))
