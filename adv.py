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
from hsja import hop_skip_jump_attack
#from demos.models import CNN



class Target(nn.Module):
    def __init__(self):
        super(Target, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3,96,3), # 96*30*30
            nn.GroupNorm(32, 96),
            nn.ELU(),
            
            nn.Dropout2d(0.2),
            
            nn.Conv2d(96, 96, 3), # 96*28*28
            nn.GroupNorm(32, 96),
            nn.ELU(),
            
            nn.Conv2d(96, 96, 3), # 96*26*26
            nn.GroupNorm(32, 96),
            nn.ELU(),
            
            nn.Dropout2d(0.5),
            
            nn.Conv2d(96, 192, 3), # 192*24*24
            nn.GroupNorm(32, 192),
            nn.ELU(),
            
            nn.Conv2d(192, 192, 3), # 192*22*22
            nn.GroupNorm(32, 192),
            nn.ELU(),
           
            nn.Dropout2d(0.5),
            
            nn.Conv2d(192, 256, 3), # 256*20*20
            nn.GroupNorm(32, 256),
            nn.ELU(),
            
            nn.Conv2d(256, 256, 1), # 256*20*20
            nn.GroupNorm(32, 256),
            nn.ELU(),
            
            nn.Conv2d(256, 10, 1), # 10*20*20
            nn.AvgPool2d(20) # 10*1*1
        )

    def forward(self,x):
        out = self.conv_layer(x)
        out = out.view(-1,10)

        return out



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
mnist_train = dsets.CIFAR10(root='./data/',
                          train=True,
                          transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)), AddGaussianNoise(0., 0.02), transforms.Normalize((0.1307,), (0.3081,)),
    ]),
                          download=True)

mnist_test = dsets.CIFAR10(root='./data/',
                         train=False,
                         transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.Normalize((0.1307,), (0.3081,))]),
                         download=True)






### The data loader part
batch_size = 128

train_loader  = torch.utils.data.DataLoader(dataset=mnist_train,
                                           batch_size=batch_size,
                                           shuffle=False,)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                         batch_size=batch_size,
                                         shuffle=False)



'''

##The most important line where you define the model
model = models.resnet18(pretrained = True).cuda()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)


n_inputs = model.fc.in_features
last_layer = nn.Linear(n_inputs, 10).cuda()
model.fc = last_layer


print(model.fc)
num_epochs  =50
acc = 0.0
for epoch in range(num_epochs):

    total_batch = len(mnist_train) // batch_size
    ct = 0
    for i, (batch_images, batch_labels) in enumerate(train_loader):
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
    if ct/len(mnist_train) > acc:
        torch.save(model, 'resnet_224_10_g02.pth')



'''

#Code for loading in trained weights
'''

'''
pretrained_weights = torch.load('resnet_224_10_nonoise.pth')
model = models.resnet18().cuda()
n_inputs = model.fc.in_features
last_layer = nn.Linear(n_inputs, 10).cuda()
model.fc = last_layer
model.load_state_dict(pretrained_weights.state_dict())
'''


'''
#Evaluation phase for the code'''
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


a = [torchattacks.Square(model, eps=8/255, n_queries=500, n_restarts=1, loss='ce'),torchattacks.DeepFool(model, steps=100),torchattacks.OnePixel(model, pixels=5, inf_batch=50)]
#a = [0]
#atk = torchattacks.MultiAttack(atks)



for i in a[:1]:
    ct = 0
    for images, labels in test_loader:
        #print("Entered batch ", ct)
        ct+=1
        images = images.cuda()
        images = hop_skip_jump_attack(model, images, 2)
        #images = i(images, labels).cuda()
        
        outputs = model(images)
    
        _, predicted = torch.max(outputs.data, 1)
    
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
    
    print('Attack is ', str(i).split("(")[0], ' and Robust accuracy: %.2f %%' % (100 * float(correct) / total))