import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import sleep
#from livelossplot import PlotLosses

from networks import Generator, Discriminator
from util import Util

class PegasusGenerator():

    def __init__(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.util = Util()

        # Load training data
        self.train_loader = self.loadTrainingData()
        self.train_iterator = iter(self.util.cycle(self.train_loader))

        # Load testing data
        self.test_loader = self.loadTestingData()
        self.train_iterator = iter(self.util.cycle(self.test_loader))

        # Init generator and discriminator
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # initialise the optimiser
        self.optimiser_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimiser_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.bce_loss = nn.BCELoss()
        self.epoch = 0
        #liveplot = PlotLosses()

    def loadTrainingData(self):
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('data', train=True, download=True, transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])), shuffle=True, batch_size=16, drop_last=True)
        return train_loader

    def loadTestingData(self):
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])), shuffle=False, batch_size=16, drop_last=True)
        return test_loader

    def train(self):
        # training loop
        while (self.epoch<25):
            print(self.epoch)
            
            # arrays for metrics
            logs = {}
            gen_loss_arr = np.zeros(0)
            dis_loss_arr = np.zeros(0)

            # iterate over some of the train dateset
            for i in range(1000):
                x,t = next(self.train_iterator)
                x,t = x.to(self.device), t.to(self.device)

                # train discriminator 
                self.optimiser_D.zero_grad()
                g = self.generator.generate(torch.randn(x.size(0), 100, 1, 1).to(self.device))
                l_r = self.bce_loss(self.discriminator.discriminate(x).mean(), torch.ones(1)[0].to(self.device)) # real -> 1
                l_f = self.bce_loss(self.discriminator.discriminate(g.detach()).mean(), torch.zeros(1)[0].to(self.device)) #  fake -> 0
                loss_d = (l_r + l_f)/2.0
                loss_d.backward()
                self.optimiser_D.step()
                
                # train generator
                self.optimiser_G.zero_grad()
                g = self.generator.generate(torch.randn(x.size(0), 100, 1, 1).to(self.device))
                loss_g = self.bce_loss(self.discriminator.discriminate(g).mean(), torch.ones(1)[0].to(self.device)) # fake -> 1
                loss_g.backward()
                self.optimiser_G.step()

                gen_loss_arr = np.append(gen_loss_arr, loss_g.item())
                dis_loss_arr = np.append(dis_loss_arr, loss_d.item())

                # plot some examples
                #plt.grid(False)
                #plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
                #plt.show()

                self.epoch = self.epoch+1

        g = self.generator.generate(torch.randn(x.size(0), 100, 1, 1).to(self.device))

        plt.grid(False)
        plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
        plt.show()

pg = PegasusGenerator()
pg.train()
