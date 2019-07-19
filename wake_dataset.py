import torch
from torch.utils.data import Dataset
import os
import pandas as pd


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

