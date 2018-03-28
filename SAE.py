import torch as th
from torch import nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):

    def __init__(self, inputDim, hiddenDim):
        super().__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.encoder = nn.Linear(inputDim, hiddenDim, bias=True)
        self.decoder = nn.Linear(hiddenDim, inputDim, bias=True)
        self.act = F.sigmoid

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
        self.en1 = encoderList[0]
        self.en2 = encoderList[1]
        self.en3 = encoderList[2]

        # for param in self.en1.parameters():
        #      param.requires_grad = False
        # for param in self.en2.parameters():
        #      param.requires_grad = False
        # for param in self.en3.parameters():
        #      param.requires_grad = False

        self.fc = nn.Linear(98, 10, bias=True)

    def forward(self, x):

        out = x
        # for i in range(len(self.encoderList)):
        #     out = self.encoderList[i](out, rep=True)
        out = self.en1(out, rep=True)
        out = self.en2(out, rep=True)
        out = self.en3(out, rep=True)
        out = self.fc(out)
        out = F.softmax(out)

        return out

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 392, bias=True)
        self.fc2 = nn.Linear(392, 196, bias=True)
        self.fc3 = nn.Linear(196, 98, bias=True)
        self.classify = nn.Linear(98, 10, bias=True)
        self.act = F.sigmoid

    def forward(self, x):

        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        out = self.classify(out)
        out = F.softmax(out)

        return out





