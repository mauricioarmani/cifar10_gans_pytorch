import torch.nn as nn
import torch.nn.functional as F
import torch


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
<<<<<<< HEAD
		self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
		self.bn1 = nn.BatchNorm2d(128)
		self.bn2 = nn.BatchNorm2d(256)
		self.fc1 = nn.Linear(4*4*256, 1)

	def forward(self, x):
		x = x.permute(0, 3, 1, 2)
		x = self.conv1(x)
		x = F.leaky_relu(x)

		x = self.conv2(x)
		x = self.bn1(x)
		x = F.leaky_relu(x)

		x = self.conv3(x)
		x = self.bn2(x)
		x = F.leaky_relu(x)

		x = x.view(-1, 4*4*256)
		x = self.fc1(x).view(-1)
		return x
		
=======
		hidden_size1 = 50
		hidden_size2 = 50
		self.fc1  = nn.Linear(32*32*3, hidden_size1)
		self.fc2  = nn.Linear(hidden_size1, hidden_size2)
		self.fc3  = nn.Linear(hidden_size2, 1)

	def forward(self, x):		
		x = x.view(-1, 32*32*3)
		x = self.fc1(x)
		x = F.leaky_relu(x, 0.2)
		x = self.fc2(x)
		x = F.leaky_relu(x, 0.2)
		x = self.fc3(x).view(-1)
		return x
		

>>>>>>> 192f635742ff3ba60a0bbc85a1da0a4bd2f16cd7
class Generator(nn.Module):
	def __init__(self, z_dim):
		super(Generator, self).__init__()
		self.fc1 = nn.Linear(100, 4*4*512)
<<<<<<< HEAD
		self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
		self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
		self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(512)
		self.bn2 = nn.BatchNorm2d(256)
		self.bn3 = nn.BatchNorm2d(128)

	def forward(self, z):
		x = self.fc1(z).view(-1, 512, 4, 4)
		x = self.bn1(x) 
		x = F.relu(x)
		x = F.leaky_relu(x)

		x = self.deconv1(x)
		x = self.bn2(x)
		x = F.leaky_relu(x)
		
		x = self.deconv2(x)
		x = self.bn3(x)
		x = F.leaky_relu(x)
		
		x = self.deconv3(x)
		x = F.tanh(x).permute(0, 2, 3, 1)
		return x
=======
		self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=2)
		self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=2)
		self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=(2,2), stride=2)

	def forward(self, z):
		x = self.fc1(z).view(-1, 512, 4, 4)
		x = self.deconv1(x)
		x = F.relu(x)
		x = self.deconv2(x)
		x = F.relu(x)
		x = self.deconv3(x)
		x = F.tanh(x)		
		return x

# class Generator(nn.Module):
# 	def __init__(self, z_dim):
# 		super(Generator, self).__init__()
# 		hidden_size1 = 200
# 		hidden_size2 = 100 
# 		self.fc1 = nn.Linear(z_dim, hidden_size1)
# 		self.fc2 = nn.Linear(hidden_size1, hidden_size2)
# 		self.fc3 = nn.Linear(hidden_size2, 32*32*3)

# 	def forward(self, z):
# 		x = self.fc1(z)
# 		x = F.relu(x)
# 		x = self.fc2(x)
# 		x = F.relu(x)
# 		x = self.fc3(x)
# 		x = F.tanh(x)
# 		return x
>>>>>>> 192f635742ff3ba60a0bbc85a1da0a4bd2f16cd7
