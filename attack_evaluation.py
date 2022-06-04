import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks
from torchattacks import PGD, FGSM
#from demos.models import CNN



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,5), # 16*24*24
            nn.ReLU(),
            nn.Conv2d(16,32,5), # 32*20*20
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 32*10*10
            nn.Conv2d(32,64,5), # 64*6*6
            nn.ReLU(),
            nn.MaxPool2d(2,2) #64*3*3
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )       
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(-1,64*3*3)
        out = self.fc_layer(out)

        return out



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
mnist_train = dsets.MNIST(root='./data/',
                          train=True,
                          transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
    AddGaussianNoise(0., 1.)]),
                          download=True)

mnist_test = dsets.MNIST(root='./data/',
                         train=False,
                         transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
                         download=True)






### The data loader part
batch_size = 128

train_loader  = torch.utils.data.DataLoader(dataset=mnist_train,
                                           batch_size=batch_size,
                                           shuffle=False,)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                         batch_size=batch_size,
                                         shuffle=False)





##The most important line where you define the model
model = CNN().cuda()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs  =5
for epoch in range(num_epochs):

    total_batch = len(mnist_train) // batch_size
    ct = 0
    for i, (batch_images, batch_labels) in enumerate(train_loader):
        #X = atk(batch_images, batch_labels).cuda()
        X = batch_images.cuda()
        Y = batch_labels.cuda()

        pre = model(X)
        #print(pre.shape,Y.shape)
        cost = loss(pre, Y)
        ct += (torch.max(pre.data, 1)[-1] == Y).sum()
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, total_batch, cost.item()))
    print(f"Accuracy at epoch {epoch} is {ct/len(mnist_train)}")

    
    
model.eval()

correct = 0
total = 0

for images, labels in test_loader:
    
    images = images.cuda()
    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))


from torchattacks import *
model.eval()

correct = 0
total = 0

atk = FGSM(model, eps=0.3)
atk1 = torchattacks.FGSM(model, eps=3)
#atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/200, steps=40, random_start=True)
atk = torchattacks.MultiAttack([atk1])

atks = [
    torchattacks.FGSM(model, eps=8/255),
    torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=100),
    torchattacks.RFGSM(model, eps=8/255, alpha=2/255, steps=100),
    torchattacks.CW(model, c=1, lr=0.01, steps=100, kappa=0),
    torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True),
    torchattacks.PGDL2(model, eps=1, alpha=0.2, steps=100),
    torchattacks.EOTPGD(model, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
    torchattacks.FFGSM(model, eps=8/255, alpha=10/255),
    torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=100),
    torchattacks.MIFGSM(model, eps=8/255, alpha=2/255, steps=100, decay=0.1),
    torchattacks.VANILA(model),
    torchattacks.GN(model, std=0.1),
    torchattacks.APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    torchattacks.APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    torchattacks.APGDT(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
    torchattacks.FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    torchattacks.FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
    torchattacks.Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
    torchattacks.AutoAttack(model, eps=8/255, n_classes=10, version='standard'),
    torchattacks.OnePixel(model, pixels=5, inf_batch=50),
    torchattacks.DeepFool(model, steps=100),
    torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
]


#atk = torchattacks.MultiAttack(atks)



for i in atks[:]:
    for images, labels in test_loader:
        images = i(images, labels).cuda()
        outputs = model(images)
    
        _, predicted = torch.max(outputs.data, 1)
    
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
    
    print('Attack is ', str(i).split("(")[0], ' and Robust accuracy: %.2f %%' % (100 * float(correct) / total))