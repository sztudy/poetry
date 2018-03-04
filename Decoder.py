# coding: utf-8

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from Attention import Attn

use_cuda = torch.cuda.is_available()
#use_cuda = False
poem_type_size = 100

with open('model/vectors.pkl','rb') as f:
    vec = pickle.load(f)

poem_five = Variable(torch.from_numpy(vec[0]).float()).view(1,1,-1)
poem_seven = Variable(torch.from_numpy(vec[1]).float()).view(1,1,-1)

if use_cuda:
    poem_five = poem_five.cuda()
    poem_seven = poem_seven.cuda()

class Decoder(nn.Module):
    def __init__(self, theme_size, wv_size, hidden_size, output_size, h2o_size, n_layers=1):
        super(Decoder, self).__init__()
        self.wv_size = wv_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        self.attn = Attn(theme_size + hidden_size, hidden_size)
        self.gru = nn.GRU(theme_size + wv_size + poem_type_size, hidden_size)
        self.h2o = nn.Linear(theme_size + wv_size + hidden_size + poem_type_size, h2o_size)
        self.out = nn.Linear(h2o_size, output_size)

    def forward(self, encoder_outputs, input_wv, last_hidden, poem_type):
        if poem_type == 5:
            poem_type_vector = poem_five
        else:
            poem_type_vector = poem_seven

        if use_cuda:
            poem_type_vector = poem_type_vector.cuda()

        attn_weights = self.attn(last_hidden[-1], encoder_outputs)# last_hidden[-1]是2维
        #print(attn_weights.size())
        #print(encoder_outputs.transpose(0, 1).size())
        theme_vector = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        theme_vector = theme_vector.transpose(0, 1) # 1 x B x N

        #形状必须是 1 x B x N
        rnn_input = torch.cat((theme_vector, input_wv), 2)
        rnn_input = torch.cat((rnn_input, poem_type_vector), 2)
        output, hidden = self.gru(rnn_input, last_hidden) 
        
        # Final output layer
        #print(output.size())
        #print(rnn_input.size())
        output = output.squeeze(0) # B x N
        output = F.log_softmax(F.relu(self.out(F.relu(self.h2o(torch.cat((output, rnn_input.squeeze(0)), 1))))), dim=1)
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

    def init_hidden(self):
        hidden0 = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

        if use_cuda:
            return hidden0.cuda()
        else:
            return hidden0

    def init_input(self):
        input0 = Variable(torch.zeros(1, 1, self.wv_size))

        if use_cuda:
            return input0.cuda()
        else:
            return input0
