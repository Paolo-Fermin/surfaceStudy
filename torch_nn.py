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

wake_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'))
print(len(wake_dataset))

data_loader = DataLoader(dataset=wake_dataset, batch_size=1, shuffle=True)

for i, sample in enumerate(data_loader):
	print(next(iter(data_loader)))


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
'''
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
'''
