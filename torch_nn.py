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

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss

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

class WakeDataset(Dataset):
	'''Dataset to manage wake image data'''

	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		print("root dir: " + str(self.root_dir))
		print(os.listdir(self.root_dir))

		#get input variable combinations and store them in a list
		#input vars are read from the case directories
		input_combos = []
		for case in os.listdir(root_dir):
			if case.startswith('dTdz'):
				input_combos.append([float(case[4:9]), float(case[-3:])])

		self.length = len(input_combos)

		#turn input combos into a tensor
		self.input_combos_tensor = torch.FloatTensor(input_combos)
			
		self.transform = transform

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		#return data in necessary format
		case_dir = '/%s/dTdz%0.3f_z%d' % (self.root_dir, self.input_combos_tensor[index][0], self.input_combos_tensor[index][1])
		uy_data = pd.read_csv(os.path.join(case_dir, 'Uy.csv'), header=None)
		#print(uy_data)		
		
		#trim data to 128x1024		
		drop_cols = 33
		uy_data.drop(uy_data.columns[:drop_cols/2], axis=1, inplace=True)
		uy_data.drop(uy_data.columns[-drop_cols/2-1:], axis=1, inplace=True)
		#print(uy_data)

		#convert data to tensor
		self.uy_data_tensor = torch.FloatTensor(uy_data.values)
		#print('Target size: ' + str(self.uy_data_tensor.size()))

		return self.input_combos_tensor[index].view(1, 1, 2), self.uy_data_tensor.view(1, 128, 1024)

print('cwd: ' + str(os.getcwd()))

train_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'))
val_dataset = WakeDataset(os.path.join(os.getcwd(), 'data', 'test_data'))
#train_dataset, val_dataset = random_split(wake_dataset, [7, 2])
#print(len(wake_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)


loss_fn = nn.MSELoss()

lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 2000
log_interval = 100

#create trainer and evaluator
trainer = create_supervised_trainer(model, optimizer, loss_fn)
evaluator = create_supervised_evaluator(model, metrics={'mse':Loss(loss_fn)})

#create visdom plots
vis = visdom.Visdom()
def create_plot_window(vis, xlabel, ylabel, title):
	return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss')
train_avg_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Avg Loss')
val_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Validation Avg Loss')

#event handlers
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
	iter = (engine.state.iteration - 1) % len(train_loader) + 1
	if iter % log_interval == 0:
		print('Epoch [{}] Loss: {:.6e}'.format(engine.state.epoch, engine.state.output))
		vis.line(X=np.array([engine.state.iteration]),
			 Y=np.array([engine.state.output]),
			 update='append', win=train_loss_window)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine): 
	evaluator.run(train_loader)
	metrics = evaluator.state.metrics
	avg_mse = metrics['mse']
	print('Training - Epoch: {} Avg Loss: {:.6e}'.format(engine.state.epoch, avg_mse))
	vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_mse]), 
		win=train_avg_loss_window, update='append')

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
	evaluator.run(val_loader)
	metrics = evaluator.state.metrics
	avg_mse = metrics['mse']
	print('Validation - Epoch: {} Avg Loss: {:.6e}\n'.format(engine.state.epoch, avg_mse))
	vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_mse]), 
		win=val_avg_loss_window, update='append')

#run training
trainer.run(train_loader, max_epochs=epochs)

#save model to a new file
i = 0
while os.path.exists('./wake_net_%d.pt' % i):
	i += 1
print('Saving model...')
torch.save(model.state_dict(), 'wake_net_%d.pt' % i)
print('Model saved')

print('Finished training')
print('Execution time: ' + str(datetime.now() - start_time))

		
