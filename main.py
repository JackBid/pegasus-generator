import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import sleep

from networks import Generator, Discriminator
from util import Util

class PegasusGenerator():

    def __init__(self):
        
        # Path for saving and loading models
        self.generatorPath = 'models/generator.pth'
        self.discriminatorPath = 'models/discriminator.pth'
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.batchSize = 32
        
        # Create generator and discriminator
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Load weights
        if self.device == torch.device('cpu'):
            self.generator.load_state_dict(torch.load(self.generatorPath, map_location=torch.device('cpu')))
            self.discriminator.load_state_dict(torch.load(self.discriminatorPath, map_location=torch.device('cpu')))
        else:
            self.generator.load_state_dict(torch.load(self.generatorPath))
            self.discriminator.load_state_dict(torch.load(self.discriminatorPath))

        self.class_names = {'airplane' : 0, 'car' : 1, 'bird' : 2, 'cat' : 3, 'deer' : 4, 'dog' : 5, 'frog' : 6, 'horse' : 7, 'ship' : 8, 'truck' : 9}
        self.util = Util()

        # Load training data
        self.trainset = self.loadTrainingData()
        self.util.filterDataset(self.trainset, [self.class_names['bird'], self.class_names['horse'], self.class_names['airplane'], self.class_names['deer']])
        self.train_loader = torch.utils.data.DataLoader(self.trainset, shuffle=True, batch_size=self.batchSize, drop_last=True)

        # Load testing data
        self.testset = self.loadTestingData()
        self.util.filterDataset(self.testset, [self.class_names['bird'], self.class_names['horse']])
        self.test_loader = torch.utils.data.DataLoader(self.testset, shuffle=True, batch_size=self.batchSize, drop_last=True)

        # initialise the optimiser
        self.optimiser_G = torch.optim.Adam(self.generator.parameters(),  lr=0.0002, betas=(0.5,0.99))
        self.optimiser_D = torch.optim.Adam(self.discriminator.parameters(),  lr=0.0002, betas=(0.5,0.99))
        self.bce_loss = nn.BCELoss()
        self.epoch = 0

    def loadTrainingData(self):
        trainset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))
        return trainset

    def loadTestingData(self):
        testset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))
        return testset

    def train(self):

        count = 0

        #g = self.generator.generate(torch.randn(self.batchSize, 100, 1, 1).to(self.device))

        #plt.grid(False)
        #plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
        #plt.show()

        for epoch in range(200):

            # arrays for metrics
            logs = {}
            gen_loss_arr = np.zeros(0)
            dis_loss_arr = np.zeros(0)

            # iterate over the training dataset
            for batch, targets in self.train_loader:

                batch, targets = batch.to(self.device), targets.to(self.device)
                
                # applying label softness
                real_label = torch.full((self.batchSize, ), 1 * (1 - random.random() * 0.3), device=self.device)
                fake_label = torch.full((self.batchSize, ), 0.3, device=self.device)

                # train discriminator 
                self.optimiser_D.zero_grad()

                # process all real batch first

                # calculate real loss
                l_r = self.bce_loss(self.discriminator.discriminate(batch), real_label) # real -> 1
                # backpropogate
                l_r.backward()
                
                # process all fake batch
                g = self.generator.generate(torch.randn(batch.size(0), 100, 1, 1).to(self.device))
                # calculate fake loss
                l_f = self.bce_loss(self.discriminator.discriminate(g), fake_label) #  fake -> 0      
                # backpropogate
                l_f.backward()

                # step optimsier
                self.optimiser_D.step()

                loss_d = (l_r + l_f) / 2
                dis_loss_arr = np.append(dis_loss_arr, loss_d.mean().item())

            count += 1

            if count == 2:
                # train generator
                self.optimiser_G.zero_grad()
                g = self.generator.generate(torch.randn(batch.size(0), 100, 1, 1).to(self.device))

                loss_g = self.bce_loss(self.discriminator.discriminate(g).view(-1), real_label) # fake -> 1

                loss_g.backward()
                self.optimiser_G.step()

                count = 0

            #en_loss_per_epoch.append(gen_loss_arr[len(gen_loss_arr) - 1])
            #dis_loss_per_epoch.append(dis_loss_arr[len(dis_loss_arr) - 1])

            print('Training epoch %d complete' % epoch)
            
            #g = self.generator.generate(torch.randn(batch.size(0), 100, 1, 1).to(self.device))

            #plt.grid(False)
            #plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
            #plt.show()

            torch.save(self.generator.state_dict(), self.generatorPath)
            torch.save(self.discriminator.state_dict(), self.discriminatorPath)

pg = PegasusGenerator()
pg.train()
