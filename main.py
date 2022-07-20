print('Bot activating...')
import time 
start = time.time()
import random
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
print('Dialog model loading...')
from Dialog_Model import get_answer
print('Dialog model ready...')
from mat import spell_check
print('Mat check ready...')
import torch
import os



from config import TOKEN
print('Cocktail model loading...')
from cocktails import cocktail
print('Cocktail model ready...')
#from model1 import pasha_technic
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

num_cocktails = 20
print(f'Generating {num_cocktails} neurococktails...')
default_cocktails = []
for i in range(num_cocktails):
    default_cocktails.append(cocktail('коктейль'))
    print(i+1, 'cocktail ready!')
print('Cocktails ready!')
end = time.time()
print(f'Bot started and it taked {(end-start):.0f}sec!\nHave fun!')
users_dict = {}

#приветствие
@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет! Будем общаться?")

#помощь
@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("Напиши мне какую-нибудь фразу , и я отпрпавлю тебе твой пиздешь в стиле Паши Техника!")

#обработка сообщения
@dp.message_handler()
async def echo_message(msg: types.Message):
    
    # print('ID', msg.from_user.id, 'send message'
    

    if msg.from_user.id not in users_dict:
        users_dict[msg.from_user.id] = torch.zeros((1, 0), dtype=torch.int)


    for k,v in users_dict.items():
        hist_length = 30
        if v.shape[1] > hist_length:
            # print(k,v)
            new = v[0][-hist_length:].unsqueeze(0)
            users_dict[k] = new

    answer = msg.text

    # print(users_dict)
    cocktail_counter = 0
    if answer.find('коктейл') != -1 or answer.find('Коктейл') != -1:
        # cock = cocktail('коктейль')
        # if len(cock) == 0:
        #     cock = cocktail('коктейль')
        # answer = 'Смотри, что у меня нашлось\n' + cock
        # cocktail_counter += 1
        #     if cocktail_counter <= 10:
        answer = 'Смотри, что у меня нашлось:\n' + random.choice(default_cocktails)
    else:
        answer, chat_history = get_answer(answer, users_dict[msg.from_user.id])
        users_dict[msg.from_user.id] = chat_history
        answer = spell_check(answer)
    await bot.send_message(msg.from_user.id, answer)

if __name__ == '__main__':
    executor.start_polling(dp)





