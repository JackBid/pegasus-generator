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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

BATCH_SIZE = 32
NUM_EPOCHS = 30
P_SWITCH = 1
DG_RATIO = 1
LABEL_SOFTNESS = 0.3
NORMALISE = False

class PegasusDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train, transform, target_transform, download)

        plane_label = 0
        bird_label = 2
        deer_label = 4
        horse_label = 7

        valid_classes = [plane_label, bird_label, deer_label, horse_label] # index of birds and horses

        pegasus_data = [self.data[i] for i in range(len(self.targets)) if self.targets[i] in valid_classes]

        # print(type(pegasus_data))
        pegasus_targets = [self.targets[i] for i in range(len(self.targets)) if self.targets[i] in valid_classes]

        self.data = pegasus_data
        self.targets = pegasus_targets

train_set = PegasusDataset('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ]))

test_set = PegasusDataset('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ]))

train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

G = Generator().to(device)
D = Discriminator().to(device)

if device == torch.device('cpu'):
    G.load_state_dict(torch.load('models/generator.pth', map_location=torch.device('cpu')))
    D.load_state_dict(torch.load('models/discriminator.pth', map_location=torch.device('cpu')))
else:
    G.load_state_dict(torch.load('models/generator.pth'))
    D.load_state_dict(torch.load('models/discriminator.pth'))

g = G.generate(torch.randn(BATCH_SIZE, 100, 1, 1).to(device))

plt.grid(False)
plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
plt.show()

# initialise the optimiser
optimiser_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5,0.99))
optimiser_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5,0.99))
bce_loss = nn.BCELoss()

gen_loss_per_epoch = []
dis_loss_per_epoch = []
# # training loop
for epoch in range(NUM_EPOCHS):
    
    # arrays for metrics
    logs = {}
    gen_loss_arr = np.zeros(0)
    dis_loss_arr = np.zeros(0)

    dg_count = 0

    # probabilistic label switching
    switch_rand = random.random()

    # iterate over the training dataset
    for batch, targets in train_loader:

        batch, targets = batch.to(device), targets.to(device)

        # applying label softness
        real_label = torch.full((BATCH_SIZE, ), 1 * (1 - random.random() *LABEL_SOFTNESS), device=device)
        fake_label = torch.full((BATCH_SIZE, ), 1 * LABEL_SOFTNESS, device=device)

        # if switching labels
        if P_SWITCH > switch_rand:
            temp = real_label.clone().detach()
            real_label = fake_label.clone().detach()
            fake_label = temp

        # train discriminator 
        optimiser_D.zero_grad()

        # process all real batch first

        # calculate real loss
        l_r = bce_loss(D.discriminate(batch), real_label) # real -> 1
        # backpropogate
        l_r.backward()
        
        # process all fake batch
        g = G.generate(torch.randn(batch.size(0), 100, 1, 1).to(device))
        # calculate fake loss
        l_f = bce_loss(D.discriminate(g), fake_label) #  fake -> 0      
        # backpropogate
        l_f.backward()

        # step optimsier
        optimiser_D.step()

        loss_d = (l_r + l_f) / 2
        dis_loss_arr = np.append(dis_loss_arr, loss_d.mean().item())

        #used for dg_ratio
        dg_count += 1
        
        #if trained discriminator enough
        if dg_count == DG_RATIO:
            # train generator
            optimiser_G.zero_grad()
            g = G.generate(torch.randn(batch.size(0), 100, 1, 1).to(device))

            loss_g = bce_loss(D.discriminate(g).view(-1), real_label) # fake -> 1

            loss_g.backward()
            optimiser_G.step()

            # append multiple to make plot easier to visualise
            for _ in range(DG_RATIO):
                gen_loss_arr = np.append(gen_loss_arr, loss_g.mean().item())

            dg_count = 0
            

    gen_loss_per_epoch.append(gen_loss_arr[len(gen_loss_arr) - 1])
    dis_loss_per_epoch.append(dis_loss_arr[len(dis_loss_arr) - 1])

    torch.save(G.state_dict(), 'models/generator.pth')
    torch.save(D.state_dict(), 'models/discriminator.pth')

    print('Training epoch %d complete' % epoch)

# display pegasus attempts

g = G.generate(torch.randn(BATCH_SIZE, 100, 1, 1).to(device))

plt.grid(False)
plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
plt.show()