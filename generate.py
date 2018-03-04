# coding: utf-8
import sys
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import gensim
import configparser

from Encoder import Encoder
from Decoder import Decoder

try:
    import ujson as json
except:
    import json

zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
use_cuda = torch.cuda.is_available()
#use_cuda = False
glove = gensim.models.KeyedVectors.load_word2vec_format("model/vectors.txt.glove", binary=False)
with open("model/characters.json", "r") as f:
    ch_index = json.load(f)

cfg = configparser.ConfigParser()
cfg.read('poem.conf')

n_iter = int(cfg.get('params','n_iter'))
wv_size = int(cfg.get('params','wv_size'))
e_hidden_size = int(cfg.get('params','e_hidden_size'))
d_hidden_size = int(cfg.get('params','d_hidden_size'))
a_hidden_size = int(cfg.get('params','a_hidden_size'))
d_linear_size = int(cfg.get('params','d_linear_size'))

encoder = Encoder(wv_size, e_hidden_size)
decoder = Decoder(2 * e_hidden_size, wv_size, d_hidden_size, len(ch_index), d_linear_size)

encoder.load_state_dict(torch.load('model/encoder.params.pkl'))
decoder.load_state_dict(torch.load('model/decoder.params.pkl'))

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

def gen_poem():
    gen_words = []
    poem_type = 7
    try:
        if sys.argv[2] == '5':
            poem_type = 5
    except:
        pass

    input_words = sys.argv[1]
    result = zhPattern.findall(input_words.strip())
    theme_words = ''
    if result:
        for words in result:
            theme_words += words

    print(theme_words)
    print(poem_type)

    input_flag = True
    for index, word in enumerate(theme_words):
        try:
            if input_flag:
                inputs = torch.from_numpy(glove[word]).view(1,1, -1)
                input_flag = False
            else:
                inputs = torch.cat((inputs, torch.from_numpy(glove[word]).view(1, 1, -1)), 0)
        except:
            print(word)

    try:
        inputs = Variable(inputs)
    except:
        raise Exception("None of the words you have entered is in the word vectors.")

    if use_cuda:
        inputs = inputs.cuda()

    encoder_hidden0 = encoder.init_hidden()
    encoder_outputs, encoder_hidden_n = encoder(inputs, encoder_hidden0)

    # 需不需要定义一个特殊的向量来表示“开始”这个概念，还是说就是用零向量？就用零向量吧，因为确实是啥也没有。
    decoder_hidden = decoder.init_hidden()
    if use_cuda:
        decoder_hidden = decoder_hidden.cuda()

    decoder_input = decoder.init_input()
    if use_cuda:
        decoder_input = decoder_input.cuda()
    for i in range(4):
        # decoder_input = decoder.init_input()
        # if use_cuda:
        #     decoder_input.cuda()
        for j in range(poem_type + 1):
            #print(decoder_hidden)
            decoder_output, decoder_hidden, attn_weights = decoder(encoder_outputs, decoder_input, decoder_hidden,
                                                                   poem_type)
            #gen_word = list(ch_index.keys())[list(ch_index.values()).index(decoder_output.data.topk(1)[1][0])]
            #print(decoder_output.data.topk(9))
            print(attn_weights)
            word_num = decoder_output.data.topk(1)[1][0]
            word_num = word_num[0]
            
            gen_word = [k for k,v in ch_index.items() if v == word_num][0]
            
            gen_words.append(gen_word)
            decoder_input = Variable(torch.from_numpy(glove[gen_word]).view(1, 1, -1))
            if use_cuda:
                decoder_input = decoder_input.cuda()

    for i in range(4):
        for j in range(poem_type + 1):
            print(gen_words[i * (poem_type + 1) + j], end='')
        print('')

if __name__ == '__main__':
    gen_poem()

