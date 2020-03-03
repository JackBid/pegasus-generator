import math
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
        self.util.filterDataset(self.trainset, [self.class_names['bird'], self.class_names['horse']])
        self.train_loader = torch.utils.data.DataLoader(self.trainset, shuffle=True, batch_size=16, drop_last=True)
        self.train_iterator = iter(self.util.cycle(self.train_loader))

        # Load testing data
        self.testset = self.loadTestingData()
        self.util.filterDataset(self.testset, [self.class_names['bird'], self.class_names['horse']])
        self.test_loader = torch.utils.data.DataLoader(self.testset, shuffle=True, batch_size=16, drop_last=True)
        self.test_iterator = iter(self.util.cycle(self.test_loader))

        # initialise the optimiser
        self.optimiser_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimiser_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
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

    def grad_penalty(self, M, real_data, fake_data, lmbda=10):
        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(self.device)
        lerp = alpha * real_data + ((1 - alpha) * fake_data)
        lerp = lerp.to(self.device)
        lerp.requires_grad = True
        lerp_d = M.discriminate(lerp)

        gradients = torch.autograd.grad(outputs=lerp_d, inputs=lerp, grad_outputs=torch.ones(lerp_d.size()).to(self.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda

    def train(self):
        # training loop
        while (self.epoch<25):
            
            # arrays for metrics
            logs = {}
            gen_loss_arr = np.zeros(0)
            dis_loss_arr = np.zeros(0)
            grad_pen_arr = np.zeros(0)


            # iterate over some of the train dateset
            for i in range(10):
                
                 # train discriminator k times
                for k in range(5):
                
                    x,t = next(self.train_iterator)
                    x,t = x.to(self.device), t.to(self.device)
                    self.optimiser_D.zero_grad()

                    g = self.generator.generate(torch.randn(x.size(0), 100, 1, 1).to(self.device))
                    l_r = self.discriminator.discriminate(x).mean()
                    l_r.backward(-1.0*torch.ones(1)[0].to(self.device)) # real -> -1
                    l_f = self.discriminator.discriminate(g.detach()).mean()
                    l_f.backward(torch.ones(1)[0].to(self.device)) #  fake -> 1
                                
                    loss_d = (l_f - l_r)
                    grad_pen = self.grad_penalty(self.discriminator, x.data, g.data, lmbda=10)
                    grad_pen.backward()
                    
                    self.optimiser_D.step()
                    
                    dis_loss_arr = np.append(dis_loss_arr, loss_d.item())
                    grad_pen_arr = np.append(grad_pen_arr, grad_pen.item())
                
                # train generator
                self.optimiser_G.zero_grad()
                g = self.generator.generate(torch.randn(x.size(0), 100, 1, 1).to(self.device))
                loss_g = self.discriminator.discriminate(g).mean()
                loss_g.backward(-1.0*torch.ones(1)[0].to(self.device)) # fake -> -1
                self.optimiser_G.step()
                
                gen_loss_arr = np.append(gen_loss_arr, -loss_g.item())

                self.epoch = self.epoch+1

        g = self.generator.generate(torch.randn(x.size(0), 100, 1, 1).to(self.device))

        plt.grid(False)
        plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
        plt.show()

        torch.save(self.generator.state_dict(), self.generatorPath)
        torch.save(self.discriminator.state_dict(), self.discriminatorPath)

pg = PegasusGenerator()
pg.train()
