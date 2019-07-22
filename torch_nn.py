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
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import MultiStepLR

from wake_model import WakeModel
from wake_dataset import WakeDataset

import visdom

torch.manual_seed(8)

model = WakeModel()

transform = transforms.Compose([
	transforms.Normalize([0.5], [0.5])
])

train_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'), transform=transform)
val_dataset = WakeDataset(os.path.join(os.getcwd(), 'data', 'val_data'), transform=transform)
#train_dataset, val_dataset = random_split(wake_dataset, [7, 2])
#print(len(wake_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)


loss_fn = nn.MSELoss()

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
epochs = 5000
log_interval = 9
checkpoint_interval = 250


#add learning rate scheduler
step_scheduler = MultiStepLR(optimizer, milestones=(10000, 75000))

#create visdom plots
vis = visdom.Visdom()
def create_plot_window(vis, xlabel, ylabel, title):
	return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss')
train_avg_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Avg Loss')
val_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Validation Avg Loss')

#training loop
for epoch in range(epochs):

	running_loss = 0.0
	running_val_loss = 0.0
	for x_batch, y_batch in train_loader:
		#set model to training mode
		model.train()	

		#zero grads
		optimizer.zero_grad()
		#forward pass
		y_pred = model(x_batch)
		#compute loss		
		loss = loss_fn(y_batch, y_pred)
		#compute gradients
		loss.backward()
		#update params and zero grads
		optimizer.step()
		
		#print stats
		running_loss += loss.item()
		if epoch % log_interval == log_interval - 1:
			avg_mse = running_loss / log_interval
			print('Training - Epoch: {} Avg Loss: {:.6e}'.format(epoch, avg_mse))
			vis.line(X=np.array([epoch]), Y=np.array([avg_mse]), 
				win=train_avg_loss_window, update='append')
			running_loss = 0.0
			print('Elapsed time: ' + str(datetime.now() - start_time))
	
	vis.line(X=np.array([epoch]), Y=np.array([loss.item()]), win=train_loss_window, update='append')	
		
	with torch.no_grad():	
		for x_val, y_val in val_loader:
			
			#set model to evaluation mode
			model.eval()
				
			y_pred = model(x_val)
			val_loss = loss_fn(y_val, y_pred)
		
			#print stats
			running_val_loss += loss.item()
			if epoch % log_interval == log_interval - 1:
				avg_mse = running_val_loss / log_interval
				print('Validation - Epoch: {} Avg Loss: {:.6e}\n'.format(epoch, avg_mse))
				vis.line(X=np.array([epoch]), Y=np.array([avg_mse]), 
					win=val_avg_loss_window, update='append')
				running_val_loss = 0.0
				print('Elapsed time: ' + str(datetime.now() - start_time))

	step_scheduler.step()

#save model to a new file
i = 0
while os.path.exists('./wake_net_%d.pt' % i):
	i += 1
print('Saving model...')
torch.save(model.state_dict(), 'wake_net_%d.pt' % i)
print('Model saved')

print('Finished training')
print('Execution time: ' + str(datetime.now() - start_time))

		
