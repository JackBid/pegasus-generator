import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import sleep
from networks import Generator
from networks import Discriminator
from dataset import PegasusDataset

BATCH_SIZE = 32
P_SWITCH = 1
DG_RATIO = 1
LABEL_SOFTNESS = 0.3
NORMALISE = False

class PegasusGenerator():
    def __init__(self, generator_path, discriminator_path):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Path for loading and saving model weights
        self.generator_path = generator_path
        self.discriminator_path = discriminator_path

        # Get training and testing data
        train_set = PegasusDataset('data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))

        test_set = PegasusDataset('data', train=False, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))

        self.train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

        # Create generator and discriminator
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)

        self.loadModels(self.generator_path, self.discriminator_path)

        # initialise the optimiser
        self.optimiser_G = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5,0.99))
        self.optimiser_D = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5,0.99))
        self.bce_loss = nn.BCELoss()

    # Load model weights
    def loadModels(self, generator_path, discriminator_path):
        if self.device == torch.device('cpu'):
            self.G.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
            self.D.load_state_dict(torch.load(discriminator_path, map_location=torch.device('cpu')))
        else:
            self.G.load_state_dict(torch.load(generator_path))
            self.D.load_state_dict(torch.load(discriminator_path))

    # Generate 16 images 
    def generateTestImages(self):
        g = self.G.generate(torch.randn(BATCH_SIZE, 100, 1, 1).to(self.device))

        plt.grid(False)
        plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
        plt.show()
    
    def train(self, num_epochs):
        gen_loss_per_epoch = []
        dis_loss_per_epoch = []

        # # training loop
        for epoch in range(num_epochs):
            
            # arrays for metrics
            logs = {}
            gen_loss_arr = np.zeros(0)
            dis_loss_arr = np.zeros(0)

            dg_count = 0

            # iterate over the training dataset
            for batch, targets in self.train_loader:

                batch, targets = batch.to(self.device), targets.to(self.device)

                # applying label softness
                real_label = torch.full((BATCH_SIZE, ), 1 * (1 - random.random() *LABEL_SOFTNESS), device=self.device)
                fake_label = torch.full((BATCH_SIZE, ), 1 * LABEL_SOFTNESS, device=self.device)

                # train discriminator 
                self.optimiser_D.zero_grad()

                # process all real batch first

                # calculate real loss
                l_r = self.bce_loss(self.D.discriminate(batch), real_label) # real -> 1
                # backpropogate
                l_r.backward()
                
                # process all fake batch
                g = self.G.generate(torch.randn(batch.size(0), 100, 1, 1).to(self.device))
                # calculate fake loss
                l_f = self.bce_loss(self.D.discriminate(g), fake_label) #  fake -> 0      
                # backpropogate
                l_f.backward()

                # step optimsier
                self.optimiser_D.step()

                loss_d = (l_r + l_f) / 2
                dis_loss_arr = np.append(dis_loss_arr, loss_d.mean().item())

                #used for dg_ratio
                dg_count += 1
                
                #if trained discriminator enough
                if dg_count == DG_RATIO:
                    # train generator
                    self.optimiser_G.zero_grad()
                    g = self.G.generate(torch.randn(batch.size(0), 100, 1, 1).to(self.device))

                    loss_g = self.bce_loss(self.D.discriminate(g).view(-1), real_label) # fake -> 1

                    loss_g.backward()
                    self.optimiser_G.step()

                    # append multiple to make plot easier to visualise
                    for _ in range(DG_RATIO):
                        gen_loss_arr = np.append(gen_loss_arr, loss_g.mean().item())

                    dg_count = 0
                    

            gen_loss_per_epoch.append(gen_loss_arr[len(gen_loss_arr) - 1])
            dis_loss_per_epoch.append(dis_loss_arr[len(dis_loss_arr) - 1])

            torch.save(self.G.state_dict(), self.generator_path)
            torch.save(self.D.state_dict(), self.discriminator_path)

            print('Training epoch %d complete' % epoch)


peg = PegasusGenerator('models/generator.pth', 'models/discriminator.pth')
peg.train(300)