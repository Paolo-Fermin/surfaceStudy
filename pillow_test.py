import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from wake_dataset import WakeDataset
from wake_model import WakeModel

#load the most recent model
i = 0
while (os.path.exists('wake_net_%d.pt' % i)):
	i += 1
i -= 1
model = WakeModel()

model.load_state_dict(torch.load('wake_net_%d.pt' % i))
#set to evaluation mode
model.eval()

#test_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'))
#test_dataloader = DataLoader(test_dataloader, batch_size=1, shuffle=True)

test_cases = [
	[0.001, -60],
	[0.005, -90],
	[0.010, -45],
	[0.010, -90]
]

test_case_dir = os.path.join('data', 'test_data')

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
