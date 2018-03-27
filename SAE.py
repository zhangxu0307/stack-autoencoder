import torch as th
from torch import nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):

    def __init__(self, inputDim, hiddenDim):
        super().__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.encoder = nn.Linear(inputDim, hiddenDim)
        self.decoder = nn.Linear(hiddenDim, inputDim)
        self.act = F.relu

    def forward(self, x, rep=False):
        hidden = self.encoder(x)
        hidden = self.act(hidden)
        if rep == False:
            out = self.decoder(hidden)
            out = self.act(out)
            return out
        else:
            return hidden


class SAE(nn.Module):

    def __init__(self, encoderList):

        super().__init__()

        self.encoderList = encoderList

        self.fc = nn.Linear(98, 10)
        self.act = F.relu

    def forward(self, x):

        out = x
        for encoder in self.encoderList:
            out = encoder(out, rep=True)
        out = self.fc(out)
        out = self.act(out)
        out = F.softmax(out)

        return out



