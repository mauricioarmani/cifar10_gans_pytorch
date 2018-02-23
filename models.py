import torch.nn as nn
import torch.nn.functional as F
import torch


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
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
		
class Generator(nn.Module):
	def __init__(self, z_dim):
		super(Generator, self).__init__()
		self.fc1 = nn.Linear(100, 4*4*512)
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
