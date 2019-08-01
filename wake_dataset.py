import torch
from torch.utils.data import Dataset
import os
import pandas as pd

#create class that inherits abstract class Dataset
'''
Dataset needs to be able to access the /data/ folder, go into each case and load each component's wake image. It needs to be able to return an image based upon an input of dTdz and d0 (and which component?).

Ideally, don't read in all the images at once, and only load them in through the __getitem__ method 
'''

class WakeDataset(Dataset):
	'''Dataset to manage wake image data'''

	def __init__(self, root_dir, data_dir, transform=None):
		self.root_dir = root_dir
		self.data_dir = data_dir
		print("root dir: " + str(self.root_dir))
		print(os.listdir(self.root_dir))

		#get all data and to get max and min input values
		all_inputs = []
		for case in os.listdir(os.path.join(root_dir, 'all_data')):
			if case.startswith('dTdz'):
				all_inputs.append([float(case[4:9]), float(case[-3:])])
		
		all_inputs = pd.DataFrame(all_inputs)
		all_inputs.columns = ['dTdz', 'depth']

		#store boundary values
		self.bounds = [[all_inputs['dTdz'].min(), all_inputs['depth'].min()], [all_inputs['dTdz'].max(), all_inputs['depth'].max()]]

		#get input variable combinations and store them in a list
		#input vars are read from the case directories
		cases = []
		for case in os.listdir(os.path.join(root_dir, data_dir)):
			if case.startswith('dTdz'):
				cases.append([float(case[4:9]), float(case[-3:])])

		self.length = len(cases)

		#turn input combos into a tensor
		self.cases = torch.FloatTensor(cases)
			
		self.transform = transform

	def __len__(self):
		return self.length

	def __getitem__(self, index):

		item = self.cases[index]

		dTdz = item[0]
		depth = item[1]

		#return data in necessary format
		case_dir = os.path.join(self.root_dir, self.data_dir, 'dTdz%0.3f_z%d' % (dTdz, depth))
		output = pd.read_csv(os.path.join(case_dir, 'Uy.csv'), header=None)
		#print(uy_data)		
		
		#trim data to 128x1024		
		output = self.crop(output, 1024)
		#print(uy_data)

		if self.transform:
			output = self.crop(output, 512)
			
		#convert output data to tensor
		output_tensor = torch.FloatTensor(output.values)
			
		#rescale output data to range (-1, 1)	
		y = self.rescale_by_value(output_tensor, torch.min(output_tensor), torch.max(output_tensor), -1, 1)

		#rescale inputs 
		dTdz = self.rescale_by_value(dTdz, self.bounds[0][0], self.bounds[1][0], -1, 1)
		depth = self.rescale_by_value(depth, self.bounds[0][1], self.bounds[1][1], -1, 1)	
		
		x = torch.stack([dTdz, depth], 0)		

		return x.view(1, 1, 2), y.view(1, 128, len(output.columns))
	
	def rescale(self, tensor, newMin, newMax): 	
		return newMin + (((tensor - torch.min(tensor)) * (newMax - newMin)) / (torch.max(tensor) - torch.min(tensor)))
		
	def rescale_by_value(self, val, oldMin, oldMax, newMin, newMax):
		return newMin + (((val - oldMin)) * (newMax - newMin)) / (oldMax - oldMin)

	def crop(self, df, num):
		df.drop(df.columns[-(len(df.columns) - num):], axis=1, inplace=True)
		return df

