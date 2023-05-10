import torch 
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#for showing the plots
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#import MNIST dataset
dataset =  torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

x, y = next(iter(train_dataloader))
print("x shape: ", x.size())
print("y labels", y)


fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].imshow(torchvision.utils.make_grid(x)[1], cmap='gray')
axs[1].imshow(torchvision.utils.make_grid(x)[2])
plt.show()

exit()

def corrupt(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + amount * noise

amount = torch.linspace(0, 1, x.shape[0])
noisy_x = corrupt(x, amount)

plt.imshow(torchvision.utils.make_grid(noisy_x)[0], cmap='gray')
plt.show()

class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2)
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act =  nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            if i < 2:
                h.append(x)
                x = self.downscale(x)
        
        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
              x += h.pop() # Fetching stored output (skip connection)
            x = self.act(l(x)) # Through the layer and the activation function
            
        return x
        
net = BasicUNet()
net.to(device)
x = torch.randn(8, 1, 28, 28)
print(net(x).shape)

total_param = sum([p.numel() for p in net.parameters()])
print('total parameters of the model:', total_param)

net = BasicUNet()
net.to(device)

EPOCHS = 10
BATCH_SIZE = 128
criterion = nn.MSELoss() #loss function
optim = torch.optim.Adam(net.parameters(), lr = 1e-3) 

train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

losses = []

for e in range(EPOCHS):
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        amount = torch.randn(x.shape[0]).to(device)
        x_noisy = corrupt(x, amount)
        pred = net(x_noisy)
        loss = criterion(pred, x)
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        losses.append(loss.item())
    
    ave_loss = sum(losses[-len(train_dataloader ):]) / len(train_dataloader)
    print(f'Finished epoch {e} Average loss for this epoch: {ave_loss}')

plt.plot(losses)
plt.ylim(0, 0.1)

             