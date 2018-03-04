# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
#use_cuda = False

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bigru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, inputs, init_hidden):
        output, hidden = self.bigru(inputs, init_hidden)

        return output, hidden

    def init_hidden(self):
        hidden0 = Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size))
        if use_cuda:
            return hidden0.cuda()
        else:
            return hidden0


