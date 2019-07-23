import torch.nn as nn


'''
class WakeModel(nn.Module):

	def __init__(self):
		super(WakeModel, self).__init__()
		
		self.conv1 = nn.ConvTranspose2d(1, 256, kernel_size=(1, 7), stride=1, padding=0)
		self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1)
		self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1)
		self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2, padding=1)
		self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=2, padding=1)
		self.conv6 = nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=2, padding=1)
		self.conv7 = nn.ConvTranspose2d(8, 4, kernel_size=(4, 4), stride=2, padding=1)
		self.conv8 = nn.ConvTranspose2d(4, 1, kernel_size=(4, 4), stride=2, padding=1)
	
	def forward(self, x):
		x = nn.PReLU()
		x = nn.InstanceNorm2d(1)
		x = nn.PReLU()
		x = nn.PReLU()
		x = nn.PReLU()
		x = nn.PReLU()
		x = nn.PReLU()
		x = nn.PReLU()			
		x = nn.Tanh()
		return x
'''

def WakeModel():
	
	model = nn.Sequential(
		nn.ConvTranspose2d(1, 1024, kernel_size=(1, 7), stride=1, padding=0),
		nn.PReLU(),
		nn.InstanceNorm2d(1),
		nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),	
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=2, padding=1),	
		nn.PReLU(),	
		nn.Dropout(p=0.1), 	
		nn.ConvTranspose2d(16, 1, kernel_size=(4, 4), stride=2, padding=1),
		nn.Tanh()
	)
	
	return model
