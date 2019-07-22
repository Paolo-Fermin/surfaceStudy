from datetime import datetime

start_time = datetime.now()

import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.dataset import random_split

from wake_model import WakeModel
from wake_dataset import WakeDataset

torch.manual_seed(8)

model = WakeModel()

train_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'))
val_dataset = WakeDataset(os.path.join(os.getcwd(), 'data', 'val_data'))
print('cwd: ' + str(os.getcwd()))

#print(len(wake_dataset))

#train_dataset, val_dataset = random_split(wake_dataset, [7, 2])

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

loss_fn = nn.MSELoss()

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 10000
print_interval = 1000

#vars for plotting loss
losses = []
val_losses = []

#training loop
for epoch in range(epochs):

	running_loss = 0.0
	running_val_loss = 0.0
	for x_batch, y_batch in train_loader:
		#set model to training mode
		model.train()	
	
		#print('x_batch = ' + str(x_batch))
		#print('y_batch = ' + str(y_batch))
		
		#zero grads
		optimizer.zero_grad()
		#forward pass
		y_pred = model(x_batch.view(1, 1, 1, 2))
		#compute loss		
		loss = loss_fn(y_batch, y_pred)
		#compute gradients
		loss.backward()
		#update params and zero grads
		optimizer.step()
		
		#losses.append(loss)
		
		#print stats
		running_loss += loss.item()
		if epoch % print_interval == print_interval - 1:
			print('[%d] training loss: %.12fE' % (epoch, running_loss / print_interval))
			running_loss = 0.0
			print('Elapsed time: ' + str(datetime.now() - start_time))

	with torch.no_grad():	
		for x_val, y_val in val_loader:
			
			#set model to evaluation mode
			model.eval()
				
			y_pred = model(x_val.view(1, 1, 1, 2))
			val_loss = loss_fn(y_val, y_pred)
			
			val_losses.append(val_loss)
		
			#print stats
			running_val_loss += loss.item()
			if epoch % print_interval == print_interval - 1:
				print('[%d] validation loss: %.12fE' % (epoch, running_val_loss / print_interval))
				running_val_loss = 0.0
				print('Elapsed time: ' + str(datetime.now() - start_time))
		
print('Finished training')
print('Execution time: ' + str(datetime.now() - start_time))

		
