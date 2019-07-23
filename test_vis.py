import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from torch.utils.data import DataLoader
from wake_dataset import WakeDataset
from wake_model import WakeModel

#load the most recent model
i = 0
while (os.path.exists('./logs/wake_net_%d' % i)):
	i += 1
i -= 1
model = WakeModel()

parser = argparse.ArgumentParser()
parser.add_argument('--num', help='specify which model number to test')
parser.add_argument('--crop', help='specify whether to crop real image for comparison', action='store_true')
args = parser.parse_args()
if args.num:
	i = int(args.num)

try:
	model.load_state_dict(torch.load('./logs/wake_net_%d/wake_net_%d_dict.pt' % (i, i)))
except RuntimeError: 
	model = torch.load('./logs/wake_net_%d/wake_net_%d_model.pt' % (i, i))
#set to evaluation mode
model.eval()

test_case_dir = os.path.join('data', 'test_data')

def crop(df):
	df.drop(df.columns[-(len(df.columns) - 512):], axis=1, inplace=True)
	return df

with torch.no_grad():
	for i, case in enumerate(os.listdir(test_case_dir)):
		
		case_vals = [float(case[4:9]), float(case[-3:])]

		wake_pred = model(torch.FloatTensor(case_vals).view(1, 1, 1, 2))
		print(wake_pred)
		print(wake_pred.size())
		wake_pred = torch.squeeze(wake_pred)
		print(wake_pred)
		print(wake_pred.size())
		#print(wake_pred_squeezed.size())

		wake_real = pd.read_csv(os.path.join(test_case_dir, case, 'Uy.csv'))
		if args.crop:
			wake_real = crop(wake_real)
		
		wake_real_np = wake_real.values
	
		fig = plt.figure(i)
		fig.suptitle(str(case))

		plt.subplot(2, 1, 1)
		plt.pcolor(wake_pred)
		plt.colorbar()

		plt.subplot(2, 1, 2)
		plt.pcolor(wake_real_np)
		plt.colorbar()

plt.show()
