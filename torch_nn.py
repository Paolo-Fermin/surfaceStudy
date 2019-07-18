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

torch.manual_seed(8)

model = nn.Sequential(
	nn.ConvTranspose2d(1, 256, kernel_size=(4, 7), stride=1, padding=0),
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

class WakeDataset(Dataset):
	'''Dataset to manage wake image data'''

	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		print("root dir: " + str(self.root_dir))
		print(os.listdir(self.root_dir))

		input_combos = []
		#get input variable combinations and store them in a list
		#input vars are read from the case directories
		for case in os.listdir(root_dir):
			if case.startswith('dTdz'):
				input_combos.append([float(case[4:9]), float(case[-3:])])

		self.length = len(input_combos)
		#assign an index to each dTdz, d0 combination	
		#for i in range(self.num_samples):

		#turn input combos into a tensor
		self.input_combos_tensor = torch.FloatTensor(input_combos)
			
		self.transform = transform

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		#return data in necessary format
		
		#create array that will store 
		#(dTdz, d0, [UySym_down], [UzSym_down])
		case_dir = '/%s/dTdz%0.3f_z%d' % (self.root_dir, self.input_combos_tensor[index][0], self.input_combos_tensor[index][1])
		uy_data = pd.read_csv(os.path.join(case_dir, 'UySym_down.csv'), index_col=0)
		self.uy_data_tensor = torch.FloatTensor(uy_data.values)
		#print(uy_data_tensor)
		#uz_data = pd.read_csv(os.path.join(case_dir, 'UzSym_down.csv'), index_col=0)
		#uz_data_tensor = torch.FloatTensor(uz_data.values)
		#print(uz_data_tensor)

		#self.images = torch.FloatTensor(uy_data.values, uz_data.values)		
		return self.input_combos_tensor[index], self.uy_data_tensor

print('cwd: ' + str(os.getcwd()))

train_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'))
test_dataset = WakeDataset(os.path.join(os.getcwd(), 'data', 'test_data'))

#print(len(wake_dataset))

#train_dataset, val_dataset = random_split(wake_dataset, [7, 2])

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

#for i, sample in enumerate(data_loader):
#	print(next(iter(data_loader)))


'''
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
'''

loss_fn = nn.MSELoss()

lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 10000
print_interval = 100

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

		
