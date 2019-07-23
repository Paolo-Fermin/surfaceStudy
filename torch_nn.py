from datetime import datetime

start_time = datetime.now()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import logging
from torchsummary import summary

from utils import set_logger

#get index of most recent model name to save to new file

i = 0
while os.path.exists(os.path.join(os.getcwd(), 'logs', 'wake_net_%d' %i)):
	i += 1
model_name = 'wake_net_%d' % i
logdir = os.path.join(os.getcwd(), 'logs', model_name)
os.mkdir(logdir)
set_logger(os.path.join(logdir, model_name + str('_training.log')))

torch.manual_seed(8)

model = WakeModel()

logging.info('cwd: ' + str(os.getcwd()))

#log model architecture
#logging.info(summary(model, (1, 1, 2)))
summary(model, (1, 1, 2))

train_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'), transform=True)
val_dataset = WakeDataset(os.path.join(os.getcwd(), 'data', 'val_data'), transform=True)
#train_dataset, val_dataset = random_split(wake_dataset, [7, 2])
#print(len(wake_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)


loss_fn = nn.MSELoss()

lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
epochs = 3000
log_interval = 1

logging.info('loss: {}'.format(loss_fn))
logging.info('starting lr: {}'.format(lr))
logging.info('epochs: {}'.format(epochs))

#add learning rate scheduler
step_scheduler = MultiStepLR(optimizer, milestones=(epochs * .3, epochs * .6, epochs * .9))

#create visdom plots
vis = visdom.Visdom()
def create_plot_window(vis, xlabel, ylabel, title):
	return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

#train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss')
train_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Training Avg Loss - %s' % model_name)
val_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Validation Avg Loss - %s' % model_name)

total_iterations = 0

#training loop
for epoch in range(epochs):

	running_loss = 0.0
	running_val_loss = 0.0
	train_iterations = 0.0 
	val_iterations = 0.0
	for x_batch, y_batch in train_loader:
		#set model to training mode
		model.train()	
		
		total_iterations += 1
		train_iterations += 1	
	
		#zero grads
		optimizer.zero_grad()
		#forward pass
		y_pred = model(x_batch)
		#compute loss		
		loss = loss_fn(y_pred, y_batch)
		
		running_loss += loss.item()		
		#print(loss.item())
		#compute gradients
		loss.backward()
		#update params and zero grads
		optimizer.step()

		#print stats
	
		#vis.line(X=np.array([total_iterations]), Y=np.array([loss.item()]), win=train_loss_window, update='append')	

	with torch.no_grad():	
		for x_val, y_val in val_loader:
	
			val_iterations += 1			

			#set model to evaluation mode
			model.eval()
				
			y_pred = model(x_val)
			val_loss = loss_fn(y_pred, y_val)
		
			#print stats
			running_val_loss += loss.item()


	if epoch % log_interval == log_interval - 1:
	
		avg_train_loss = running_loss / train_iterations
		logging.info('Training - Epoch: {} Avg Loss: {:.6e}'.format(epoch, avg_train_loss))
		vis.line(X=np.array([epoch]), Y=np.array([avg_train_loss]), 
			win=train_avg_loss_window, update='append')
		running_loss = 0.0
		train_iterations = 0.0
		
		avg_val_loss = running_val_loss / val_iterations
		logging.info('Validation - Epoch: {} Avg Loss: {:.6e}\n'.format(epoch, avg_val_loss))
		vis.line(X=np.array([epoch]), Y=np.array([avg_val_loss]), 
			win=val_avg_loss_window, update='append')
		running_val_loss = 0.0
		val_iterations = 0.0
		logging.info('Elapsed time: ' + str(datetime.now() - start_time))

	step_scheduler.step()

#save model to a new file
logging.info('Saving state dict...')
torch.save(model.state_dict(), os.path.join(logdir, model_name + '_dict.pt'))
logging.info('State dict saved')

torch.save(model, os.path.join(logdir, model_name + '_model.pt'))

logging.info('Finished training')
logging.info('Execution time: ' + str(datetime.now() - start_time))

		
