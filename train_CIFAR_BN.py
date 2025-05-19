# classification mnist example
import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init

from models.ResNet_CIFAR_BN import ResNet_20_CIFAR_BN


from jittor.dataset.mnist import MNIST
from jittor.dataset.cifar import CIFAR10

import jittor.transform as trans

jt.flags.use_cuda = 1 # if jt.flags.use_cuda = 1 will use gpu

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs.shape -- [64,32,32,3,]
        # targets.shape -- [64,]
        # import pdb; pdb.set_trace()
        inputs = inputs.permute(0, 3, 1, 2).float32()
        # inputs.shape -- [64,3,32,32,]

        # print(f'inputs.shape is {inputs.shape}')
        
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step (loss)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0]))


def test(model, val_loader, epoch):
    model.eval()

    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.permute(0, 3, 1, 2).float32()
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}'.format(epoch, \
                    batch_idx, len(val_loader),100. * float(batch_idx) / len(val_loader), acc))
    print ('Total test acc =', total_acc / total_num)
    # Total test acc = 0.5689


def main ():
    batch_size = 64
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 1
    # train_loader = MNIST(train=True, transform=trans.Resize(28)).set_attrs(batch_size=batch_size, shuffle=True)
    # val_loader = MNIST(train=False, transform=trans.Resize(28)).set_attrs(batch_size=1, shuffle=False)

    train_loader = CIFAR10(train=True).set_attrs(batch_size=batch_size, shuffle=True)
    val_loader = CIFAR10(train=False).set_attrs(batch_size=1, shuffle=False)
    

    model = ResNet_20_CIFAR_BN(w=32, h=32, c=3, num_classes=10)
    

    # import pdb; pdb.set_trace()
    
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)
        test(model, val_loader, epoch)



if __name__ == '__main__':
    main()












