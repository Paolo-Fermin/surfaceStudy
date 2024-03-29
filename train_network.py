import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import MultiStepLR
from torchsummary import summary

from wake_model import WakeModelFull, WakeModelCropped
from wake_dataset import WakeDataset

import visdom
import logging
from utils import set_logger
from datetime import datetime

start_time = datetime.now()

#parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='Specify the number of epochs for training')
parser.add_argument('--crop', help='Boolean: whether to crop data by half (default False)', action='store_true')
args = parser.parse_args()
if args.epochs:
	epochs = int(args.epochs)
else: 
	epochs = 1000
if args.crop:
	crop = args.crop
else:
	crop = False

#define random torch seed for consistency
torch.manual_seed(8)

#get index of most recent model name to save to new file
i = 0
while os.path.exists(os.path.join(os.getcwd(), 'logs', 'wake_net_%d' %i)):
	i += 1
model_name = 'wake_net_%d' % i
logdir = os.path.join(os.getcwd(), 'logs', model_name)
os.mkdir(logdir)
set_logger(os.path.join(logdir, model_name + str('_training.log')))

#get model architecture
if crop:
	model = WakeModelCropped()
else:
	model = WakeModelFull()

#print model architecture
#logging.info(summary(model, (1, 1, 2)))
summary(model, (1, 1, 2))

#get pytorch datasets and create dataloaders for them
train_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'), 'train_data', transform=crop)
val_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'), 'val_data', transform=crop)

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

#define some hyperparameters and log them
loss_fn = nn.MSELoss()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
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

train_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Training Avg Loss - %s' % model_name)
val_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Validation Avg Loss - %s' % model_name)

total_iterations = 0

def save_checkpoint(state, is_best, filename=os.path.join(os.getcwd(),'logs', model_name + str('_best.pt'))):
	if is_best:
		logging.info('=> Saving a new best\n')
		torch.save(state, filename)
	else:
		logging.info('=> Validation loss did not improve\n')

best_val_loss = 1.0

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
		logging.info('Validation - Epoch: {} Avg Loss: {:.6e}'.format(epoch, avg_val_loss))
		vis.line(X=np.array([epoch]), Y=np.array([avg_val_loss]), 
			win=val_avg_loss_window, update='append')
		running_val_loss = 0.0
		val_iterations = 0.0
		logging.info('Elapsed time: ' + str(datetime.now() - start_time))

		if avg_val_loss < best_val_loss:
			is_best = True
			best_val_loss = avg_val_loss
		else:
			is_best = False	
	
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_val_loss': avg_val_loss
		}, is_best)

	step_scheduler.step()

#save model to a new file
logging.info('Saving state dict...')
torch.save(model.state_dict(), os.path.join(logdir, model_name + '_last.pt'))
logging.info('State dict saved')

torch.save(model, os.path.join(logdir, model_name + '_fullmodel.pt'))

logging.info('Finished training')
logging.info('Execution time: ' + str(datetime.now() - start_time))

		
