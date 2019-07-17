import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Dataloader
from torchvision import transforms, utils

#ignore warnings
warnings.filterwarnings('ignore')

#create class that inherits abstract class Dataset
'''
Dataset needs to be able to access the /data/ folder, go into each case and load each component's wake image. It needs to be able to return an image based upon an input of dTdz and d0 (and which component?).

Ideally, don't read in all the images at once, and only load them in through the __getitem__ method 
'''

class WakeDataset(Dataset):
	'''Dataset to manage wake image data'''

	def __init__(self, root_dir, transform=None):
	'''
	args:
		root_dir (string): directory with all images
		transform (callable, optional): optional transform to be applied on a sample
	
	'''
		self.root_dir = root_dir
		self.num_samples = sum(os.path.isdir(i) for i in os.listdir(self.root_dir))
		
		self.input_combos = []
		#get input variable combinations and store them in a list
		for i, case in enumerate(filter(os.path.isdir, os.listdir(os.getcwd()))):
			self.input_combos.append([float(case[4:9]), float(case[-3:])])
	
		#assign an index to each dTdz, d0 combination	
		#for i in range(self.num_samples):

		#turn input combos into a tensor
		#input_combos_tensor = torch.FloatTensor(input_combos)
			
		self.transform = transform

	def __len__(self):
		return self.num_samples

	def __getitem__(self, index):
		#return data in necessary format
		
		#create array that will store 
		#(dTdz, d0, [UySym_down], [UzSym_down])
		case_dir = '/%s/dTdz%0.3f_z%d' % (root_dir, self.input_combos[index][0], self.input_combos[index][1])
		uy_data = pd.read_csv(os.path.join(case_dir, 'UySym_down.csv'), index_col=0)
		self.uy_data_tensor = torch.FloatTensor(uy_data.values)
		#print(uy_data_tensor)
		#uz_data = pd.read_csv(os.path.join(case_dir, 'UzSym_down.csv'), index_col=0)
		#uz_data_tensor = torch.FloatTensor(uz_data.values)
		#print(uz_data_tensor)

		#self.images = torch.FloatTensor(uy_data.values, uz_data.values)		
	
		return self.input_combos, self.uy_data_tensor

