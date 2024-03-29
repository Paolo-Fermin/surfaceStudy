import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch.utils.data import DataLoader
from wake_dataset import WakeDataset
from wake_model import WakeModelFull, WakeModelCropped

from utils import rescale_by_value

#load the most recent model
i = 0
while (os.path.exists('./logs/wake_net_%d' % i)):
	i += 1
i -= 1
model = WakeModelFull()

#parse command line arguments, if available
parser = argparse.ArgumentParser()
parser.add_argument('--num', help='specify which model number to test')
parser.add_argument('--crop', help='specify whether to crop real image for comparison', action='store_true')
parser.add_argument('--last', help='specify whether to save the last or best model, default best', action='store_true')
args = parser.parse_args()
if args.num:
	i = int(args.num)
if args.last:
	model_option = 'last'
else:
	model_option = 'best'

#PyTorch stores models in two ways, saving just the weights or saving both the architecture and the weights. The former is preferred. See more in the Pytorch website 'Saving and Loading Models'. 
try:
	model.load_state_dict(torch.load('./logs/wake_net_%d/wake_net_%d_%s.pt' % (i, i, model_option)))
except RuntimeError: 
	model = torch.load('./logs/wake_net_%d/wake_net_%d_fullmodel.pt' % (i, i))
except IOError:
	#account for old naming convention 
	model.load_state_dict(torch.load('./logs/wake_net_%d/wake_net_%d_dict.pt' % (i, i)))

#set to evaluation mode
model.eval()

#get all data and to get max and min input values
all_inputs = []
for case in os.listdir(os.path.join(os.getcwd(), 'data', 'all_data')):
	if case.startswith('dTdz'):
		all_inputs.append([float(case[4:9]), float(case[-3:])])

all_inputs = pd.DataFrame(all_inputs)
all_inputs.columns = ['dTdz', 'depth']

#store boundary values
bounds = [[all_inputs['dTdz'].min(), all_inputs['depth'].min()], [all_inputs['dTdz'].max(), all_inputs['depth'].max()]]

#get all test case vals and store in dataframe
cases = []
test_case_dir = os.path.join('data', 'test_data')
for case in os.listdir(test_case_dir):
	if case.startswith('dTdz'):
		cases.append([float(case[4:9]), float(case[-3:])])

case_frame = pd.DataFrame(cases)
case_frame.columns = ['dTdz', 'depth']

print(case_frame)

def crop(df, num):
	df.drop(df.columns[-(len(df.columns) - num):], axis=1, inplace=True)
	return df


#get x and y axis values for accurate plotting
x_axis = [i for i in range(128)]
y_axis = [i for i in range(1024)]

for i, x_val in enumerate(x_axis):
    x_axis[i] = rescale_by_value(x_val, 0, len(x_axis), 0, 5 * 3810)

for i, y_val in enumerate(y_axis):
    y_axis[i] = rescale_by_value(float(y_val), 0, len(y_axis), 0, 750)


with torch.no_grad():
	for i, case in enumerate(cases):
	
		dTdz = float(case[0])
		depth = float(case[1])			

		#rescale inputs
		dTdz = rescale_by_value(dTdz, bounds[0][0], bounds[1][0], -1, 1)
		depth = rescale_by_value(depth, bounds[0][1], bounds[1][1], -1, 1)

		case_vals = [dTdz, depth]

		wake_pred = model(torch.FloatTensor(case_vals).view(1, 1, 1, 2))
		#print(wake_pred)
		#print(wake_pred.size())
		wake_pred = torch.squeeze(wake_pred)
		wake_pred = torch.transpose(wake_pred, 0, 1)
		#print(wake_pred)
		print(wake_pred.size())
		#print(wake_pred_squeezed.size())

		wake_real = pd.read_csv(os.path.join(test_case_dir, 'dTdz%0.3f_z%d' % (case[0], case[1]), 'Uy.csv'), header=None)
		
		#trim to 128x1024
		wake_real = crop(wake_real, 1024)		
		
		if args.crop:
			#trim to 128x512
			wake_real = crop(wake_real, 512)

		wake_real = wake_real.transpose()
		
		#convert dataframe to numpy array
		wake_real = wake_real.values
		#rescale numpy array to (-1, 1)	
		wake_real = rescale_by_value(wake_real, np.min(wake_real), np.max(wake_real), -1, 1)

		print(wake_real.shape)		

		fig, axs = plt.subplots(2)

		fig.tight_layout(h_pad=1.5)
		axs[0].set_title('Real image')
		real_plot = axs[0].pcolor(x_axis, y_axis, wake_real, vmin=-1, vmax=1)

		axs[1].set_title('Predicted image')
		pred_plot = axs[1].pcolor(x_axis, y_axis, wake_pred, vmin=-1, vmax=1)
	

		fig.subplots_adjust(right=0.8, top=0.87)
		fig.suptitle(str(case))	
		cbar_ax = fig.add_axes([.85, .15, 0.05, 0.7])
		fig.colorbar(real_plot, cax=cbar_ax)
		

		#print('Average difference: {}'.format(diff.mean()))	

plt.show()
