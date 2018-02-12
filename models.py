import torch
import torch.nn as nn

class Model(nn.Module):
	def __init__(self, nb_classes, batch_size):
		super(Model, self).__init__()
		hidden_size1 = 100
		hidden_size2 = 50
		self.fc1  = nn.Linear(32*32*3, hidden_size1)
		self.relu = nn.ReLU()
		self.fc2  = nn.Linear(hidden_size1, hidden_size2)
		self.fc3  = nn.Linear(hidden_size2, nb_classes)
		self.batch_size = batch_size
	def forward(self, x):
		batch_size = x.shape[0]
		x = x.view(batch_size, 32*32*3)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		return x


class Generator(nn.Module):
	def __init__(self, nb_classes, batch_size):
		super(Model, self).__init__()
		hidden_size1 = 64
		hidden_size2 = 64
		self.fc1  = nn.Linear(32*32*3, hidden_size1)
		self.relu = nn.ReLU()
		self.fc2  = nn.Linear(hidden_size1, hidden_size2)
		self.fc3  = nn.Linear(hidden_size2, 32*32*3)
		self.batch_size = batch_size
		
	def forward(self):
		z = torch.Tensor(self.batch_size, 32*32*3).uniform_(0, 1)
		x = self.fc1(z)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		return x
