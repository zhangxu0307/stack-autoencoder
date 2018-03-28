import torch as th
import torchvision
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
from SAE import *


def loadMNIST(batchSize):

    root = "./data/"

    trans= transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root=root, train=False, transform=trans)

    train_loader = th.utils.data.DataLoader(dataset=train_set, batch_size=batchSize, shuffle=True)
    test_loader = th.utils.data.DataLoader(dataset=test_set, batch_size=batchSize, shuffle=False)

    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total testing batch number: {}'.format(len(test_loader)))

    return train_loader, test_loader


def trainAE(encoderList, trainLayer, batchSize, epoch, useCuda = False):

    if useCuda:
        for i in range(len(encoderList)):
            encoderList[i].cuda()

    optimizer = optim.SGD(encoderList[trainLayer].parameters(), lr=0.1)
    ceriation = nn.L1Loss()
    trainLoader, testLoader = loadMNIST(batchSize=batchSize)

    for i in range(epoch):

        sum_loss = 0

        if trainLayer != 0: # 单独处理第0层，因为第一个编码器之前没有前驱的编码器了
            for i in range(trainLayer): # 冻结要训练前面的所有参数
                for param in encoderList[i].parameters():
                    param.requires_grad = False

        for batch_idx, (x, target) in enumerate(trainLoader):
            optimizer.zero_grad()
            if useCuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            x = x.view(-1, 784)
            # 产生需要训练层的输入数据
            out = x
            if trainLayer != 0:
                for i in range(trainLayer):
                    out = encoderList[i](out, rep=True)

            # 训练指定的自编码器
            pred = encoderList[trainLayer](out, rep=False)

            loss = ceriation(pred, out)
            sum_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> train layer:{}, epoch: {}, batch index: {}, train loss: {:.6f}'
                      .format(trainLayer, i, batch_idx + 1, sum_loss/batch_idx))


def trainClassifier(model, batchSize, epoch, useCuda = False):

    if useCuda:
        model = model.cuda()

    # 解锁参数
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    ceriation = nn.NLLLoss()
    trainLoader, testLoader = loadMNIST(batchSize=batchSize)

    for i in range(epoch):

        # trainning
        sum_loss = 0

        for batch_idx, (x, target) in enumerate(trainLoader):
            optimizer.zero_grad()
            if useCuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            x = x.view(-1, 784)

            out = model(x)

            loss = ceriation(out, target)
            sum_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format( i, batch_idx + 1, sum_loss/batch_idx))

        # testing
        correct_cnt, sum_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(testLoader):

            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            if useCuda:
                x, target = x.cuda(), target.cuda()
            x = x.view(-1, 784)

            out = model(x)
            loss = ceriation(out, target)
            _, pred_label = th.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()

            # smooth average
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(testLoader):
                print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    i, batch_idx + 1, sum_loss/batch_idx, correct_cnt * 1.0 / total_cnt))

if __name__ == '__main__':

    batchSize = 128
    AEepoch = 20
    epoch = 10

    encoder1 = AutoEncoder(784, 256)
    encoder2 = AutoEncoder(256, 64)
    #encoder3 = AutoEncoder(196, 98)

    encoderList = [encoder1, encoder2,]

    trainAE(encoderList, 0, batchSize, AEepoch, useCuda=True)
    trainAE(encoderList, 1, batchSize, AEepoch, useCuda=True)
    #trainAE(encoderList, 2, batchSize, AEepoch, useCuda=True)

    model = SAE(encoderList)
    #model = MLP()
    trainClassifier(model, batchSize, epoch, useCuda=True)
