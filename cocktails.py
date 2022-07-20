import numpy as np
import pandas as pd
import re
import random

# import torch
from tqdm.notebook import tqdm
# import transformers
from torch.optim import AdamW

# Загружаем токенайзер модели
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

# Эту модель подгрузим и далее обучим
# model = GPT2LMHeadModel.from_pretrained(
#     'sberbank-ai/rugpt3small_based_on_gpt2',
#     output_attentions = False,
#     output_hidden_states = False,
# )

# state_dict = torch.load(
#     '/home/igor/Downloads/NPL_Project/Project_NLP/Project_NLP/Cocktails/cocktails.pth', 
#     map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)
 
# model.save_pretrained('/home/igor/Downloads/NPL_Project/Project_NLP/Project_NLP/Cocktails')

model = GPT2LMHeadModel.from_pretrained('/home/igor/Downloads/NPL_Project/Project_NLP/Project_NLP/Cocktails')

import textwrap

# дообученная модель
def cocktail(prompt):
    prompt = tokenizer.encode(prompt, return_tensors='pt')
    temp = random.uniform(3.5, 4.5)
    max_len = random.randint(90, 110)
    rand_beams = random.randint(4, 7)
    # print('temp:', temp)
    # print('max_len:', max_len)
    # print('rand_beams:', rand_beams)
    out = model.generate(
        input_ids=prompt,
        max_length=max_len,
        num_beams=rand_beams,
        do_sample=True,
        temperature=temp,
        top_k=30,
        top_p=0.7,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        ).cpu().numpy()
    for out_ in out:
        out_str = tokenizer.decode(out_)
        index = out_str.rfind('.')
        p = textwrap.fill(out_str[:index+1], 120)
    return p