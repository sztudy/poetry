# coding: utf-8

import os
import re
import time
import numpy
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
encoder_lr = float(cfg.get('params','encoder_lr'))
decoder_lr = float(cfg.get('params','decoder_lr'))
momentum_rate = float(cfg.get('params','momentum_rate'))

encoder = Encoder(wv_size, e_hidden_size)
decoder = Decoder(2 * e_hidden_size, wv_size, d_hidden_size, len(ch_index), d_linear_size)

try:
    encoder.load_state_dict(torch.load('model/encoder.params.pkl'))
    decoder.load_state_dict(torch.load('model/decoder.params.pkl'))
    print('load')
except:
    pass


if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

criterion = nn.NLLLoss()
encoder_optimizer = optim.SGD(encoder.parameters(), lr=encoder_lr, momentum=momentum_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=decoder_lr, momentum=momentum_rate)
#encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
#decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

def train(keywords, sentences):
    poem_type = len(sentences[0]) - 1

    for index, word in enumerate(keywords):
        if index == 0:
            inputs = torch.from_numpy(glove[word]).view(1,1, -1)
        else:
            inputs = torch.cat((inputs, torch.from_numpy(glove[word]).view(1, 1, -1)), 0)

    #print("inputs size")
    #print(inputs.size())
    inputs = Variable(inputs)
    if use_cuda:
        inputs = inputs.cuda()

    encoder_hidden0 = encoder.init_hidden()
    if use_cuda:
        encoder_hidden0 = encoder_hidden0.cuda()

    encoder_outputs, encoder_hidden_n = encoder(inputs, encoder_hidden0)

    # 需不需要定义一个特殊的向量来表示“开始”这个概念，还是说就是用零向量？就用零向量吧，因为确实是啥也没有。
    decoder_hidden = decoder.init_hidden()
    if use_cuda:
        decoder_hidden = decoder_hidden.cuda()

    decoder_wv = decoder.init_input()
    if use_cuda:
        decoder_wv = decoder_wv.cuda()

    for sen in sentences:
        # decoder_wv = decoder.init_input()
        # if use_cuda:
        #     decoder_wv.cuda()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0
        for word in sen:
            decoder_output, decoder_hidden, attn_weights = decoder(encoder_outputs, decoder_wv, decoder_hidden, poem_type)
            target = Variable(torch.LongTensor([ch_index[word]]))
            if use_cuda:
                target = target.cuda()
            loss += criterion(decoder_output, target)
            decoder_wv = Variable(torch.from_numpy(glove[word]).view(1, 1, -1))
            if use_cuda:
                decoder_wv = decoder_wv.cuda()

        loss.backward(retain_graph=True)
        encoder_optimizer.step()
        decoder_optimizer.step()
        print("%s, %d, %f"%(sen, poem_type, loss/poem_type))


def train_flow(iteration):
    dirs = os.listdir("data/filtered_json")
    start = time.time()
    counter = 0
    end = False
    for i in range(iteration):
        if end:
            break
        for files in dirs:
            if end:
                break
            with open("data/filtered_json/" + files, "r", encoding='utf-8') as f:
                poems = json.load(f)
                p_num = len(poems)
                for i in range(p_num):
                    poem = poems[numpy.random.randint(p_num)]
                #for poem in poems:
                    #if time.time() - start > 37000:
                        #end = True
                        #break
                    counter += 1
                    train_flag = True
                    sentences = re.split(r'[，。！？]', poem["paragraphs"])
                    sentences = [sen + '，' for sen in sentences if len(sen) > 0]
                    keywords = poem["title"] + sentences[0][0:-1]
                    
                    if len(sentences) < 9:
                        for sen in sentences:
                            #print(sen)
                            if not train_flag:
                                break
                            if len(sen) != 6 and len(sen) !=8:
                                train_flag = False

                            for word in sen:
                                try:
                                    vec = glove[word]
                                except:
                                    train_flag = False
                                    break
                    else:
                        train_flag = False

                    if train_flag:
                        print(keywords)
                        train(keywords, sentences)
                        print('')

            torch.save(encoder.state_dict(), "model/encoder.params.pkl")
            torch.save(decoder.state_dict(), "model/decoder.params.pkl")
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

if __name__ == '__main__':
    #train_flow(n_iter)
    train_flow(100)

