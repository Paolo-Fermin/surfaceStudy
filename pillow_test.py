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

model = WakeModel()

model.load_state_dict(torch.load('wake_net.pt'))
#set to evaluation mode
model.eval()

#test_dataset = WakeDataset(os.path.join(os.getcwd(), 'data'))
#test_dataloader = DataLoader(test_dataloader, batch_size=1, shuffle=True)

test_cases = [
	[0.010, -75],
	[0.001, -75],
	[0.010, -45]
]

with torch.no_grad():
	for case in test_cases:
		wake_pred = model(torch.Tensor(case).view(1, 1, 1, 2))
		print(wake_pred)
		print(wake_pred.size())
		wake_pred = torch.squeeze(wake_pred)
		print(wake_pred)
		print(wake_pred.size())
		#print(wake_pred_squeezed.size())
		plt.pcolor(wake_pred)
		plt.colorbar()
		plt.show()
