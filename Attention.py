# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class Attn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        #hidden = hidden.unsqueeze(0)
        #hiddens = hidden.expand(max_len, -1, -1)
        #inputs = torch.cat((encoder_outputs, hiddens), 2)

        for i in range(max_len):
            output = F.relu(self.linear(torch.cat((encoder_outputs[i], hidden), 1)))
            #output = F.relu(self.linear1(output))
            output = self.attn(output)
            if i == 0:
                attn_energies = output.clone()
            else:
                attn_energies = torch.cat((attn_energies, output), 1)

        # 这里只能用softmax不能用log_softmax，因为你是要算概率，而不是要做反向传播
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

