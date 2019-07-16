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
		self.transform = transform

	def __len__(self):
		#return 2 times the number of directories in the root directory 
		return 2 * sum(os.path.isdir(i) for i in os.listdir(self.root_dir))

	def __getitem__(self, dTdz, d0):
		#return data in necessary format
		
		#create array that will store 
		#(dTdz, d0, [UySym_down], [UzSym_down])
		case_dirs = [f for f in os.listdir(root_dir) if f.startswith('dTdz')
		for case in case_dirs:
			self.
		
		return self.wake_image

	
	#function to downsample data to be 128x256
	def downsample(self, dire, component_file):
		
		#recombine file with directory path 
		component_filepath = os.path.join(dire, component_file)
		print(component_filepath)
		results = pd.read_csv(component_filepath)
		component_name = os.path.splitext(component_filepath)
		print(component_name)

		results_down = results.iloc[:, ::8] #get every nth value 
		results_down.drop(results_down.columns[-9:], axis=1, inplace=True) #drop last 9 columns to get even 256
		results_down.to_csv('%s_down.csv' % component_name[0])
			
		
