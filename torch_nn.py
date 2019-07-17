import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Dataloader
from torchvision import transforms, utils

torch.manual_seed(8)

model = nn.Sequential(
	nn.ConvTranspose2d(1, 256, kernel_size=(7, 4), stride=1, padding=0),
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
	nn.ConvTranspose2d(16, 1, kernel_size=(4, 4), stride=2, padding=1),		
	nn.Tanh()
)

#create training dataset
temps = [0.005, 0.001, 0.010]
depths = [-30, -60, -90]

for temp in temps:
	for depth in depths:
		data = [[temp], [depth]]
		in_data = torch.tensor(data)
		print(in_data)
		print(in_data.size())
		print("View = " + str(in_data.view(1, 2, 1)))

		summary(model, input_size=in_data.view(1, 2, 1).size())

loss_fn = nn.MSELoss()

lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 10000

#training loop
for epoch in range(epochs):
	
	#set model to training mode
	model.train()	

	#forward pass
	y_pred = model(x)
