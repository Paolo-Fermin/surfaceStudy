import torch
import torch.nn as nn
import os

#load the most recent model
i = 0
while (os.path.exists('wake_net_%d.pt' % i)):
	i += 1


model = nn.Sequential(
	nn.ConvTranspose2d(1, 256, kernel_size=(1, 7), stride=1, padding=0),
	nn.PReLU(),
	nn.InstanceNorm2d(1),
	nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1),
	nn.PReLU(),
	nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1),
	nn.PReLU(),
	nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2, padding=1),
	nn.PReLU(),
	nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=2, padding=1),
	nn.PReLU(),	
	nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=2, padding=1),
	nn.PReLU(),
	nn.ConvTranspose2d(8, 4, kernel_size=(4, 4), stride=2, padding=1),	
	nn.PReLU(),		
	nn.ConvTranspose2d(4, 1, kernel_size=(4, 4), stride=2, padding=1),
	nn.Tanh()
)

model.load_state_dict(torch.load('wake_net.pt'))
#set to evaluation mode
model.eval()


print(model)
