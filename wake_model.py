import torch.nn as nn

def WakeModelFull():

	model = nn.Sequential(
		nn.ConvTranspose2d(1, 256, kernel_size=(1, 7), stride=1, padding=0),
		nn.PReLU(),
		nn.InstanceNorm2d(1),
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
		nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(8, 4, kernel_size=(4, 4), stride=2, padding=1),	
		nn.PReLU(),		
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(4, 1, kernel_size=(4, 4), stride=2, padding=1),
		nn.Tanh()
	)

	return model

def WakeModelCropped():
	
	model = nn.Sequential(
		nn.ConvTranspose2d(1, 64, kernel_size=(2, 7), stride=1, padding=0),
		nn.PReLU(),
		nn.InstanceNorm2d(1),
		nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),	
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(8, 4, kernel_size=(4, 4), stride=2, padding=1),
		nn.PReLU(),
		nn.Dropout(p=0.1), 
		nn.ConvTranspose2d(4, 2, kernel_size=(4, 4), stride=2, padding=1),	
		nn.PReLU(),	
		nn.Dropout(p=0.1),
		nn.ConvTranspose2d(2, 1, kernel_size=(4, 4), stride=2, padding=1),
		nn.Tanh()
	)

	return model
