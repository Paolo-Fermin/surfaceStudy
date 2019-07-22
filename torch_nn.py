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
from wake_dataset import WakeDataset

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.param_scheduler import LRScheduler

import visdom

torch.manual_seed(8)

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

print('cwd: ' + str(os.getcwd()))

train_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'))
val_dataset = WakeDataset(os.path.join(os.getcwd(), 'data', 'val_data'))
#train_dataset, val_dataset = random_split(wake_dataset, [7, 2])
#print(len(wake_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)


loss_fn = nn.MSELoss()

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
epochs = 100000
log_interval = 1000
checkpoint_interval = 250

#create trainer and evaluator
trainer = create_supervised_trainer(model, optimizer, loss_fn)
evaluator = create_supervised_evaluator(model, metrics={'mse':Loss(loss_fn)})

#add checkpoints
checkpoint_dir = 'checkpoints'
checkpointer = ModelCheckpoint(checkpoint_dir, 'wake_model_checkpoint', save_interval=250, 		create_dir=True)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'mymodel':model})

#add learning rate scheduler
step_scheduler = MultiStepLR(optimizer, milestones=(10000, 75000))
#wrap in ignite class
scheduler = LRScheduler(step_scheduler)
trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

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
		
		#print stats
		running_loss += loss.item()
		if epoch % log_interval == log_interval - 1:
			avg_mse = running_loss / log_interval
			print('Training - Epoch: {} Avg Loss: {:.6e}'.format(epoch, avg_mse))
			vis.line(X=np.array([epoch]), Y=np.array([avg_mse]), 
				win=train_avg_loss_window, update='append')
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
				avg_mse = running_val_loss / log_interval
				print('Validation - Epoch: {} Avg Loss: {:.6e}\n'.format(epoch, avg_mse))
				vis.line(X=np.array([epoch]), Y=np.array([avg_mse]), 
					win=val_avg_loss_window, update='append')
				running_val_loss = 0.0
				print('Elapsed time: ' + str(datetime.now() - start_time))

#save model to a new file
i = 0
while os.path.exists('./wake_net_%d.pt' % i):
	i += 1
print('Saving model...')
torch.save(model.state_dict(), 'wake_net_%d.pt' % i)
print('Model saved')

print('Finished training')
print('Execution time: ' + str(datetime.now() - start_time))

		
