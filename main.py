from loader import load_CIFAR10, load_data
from models import Discriminator, Generator
from run import run
import torch


X_train, Y_train = load_CIFAR10()

X_train = X_train/127.5 - 1

BATCH_SIZE = 128
x_train_data = load_data(X_train, Y_train, BATCH_SIZE)

Z_DIM = 100
G = Generator(Z_DIM).cuda()
D = Discriminator().cuda()

z_val_ = torch.Tensor(BATCH_SIZE, 100).normal_(0,1)

epochs = 100
for epoch in range(epochs):
	print 'epoch: %d/%d' % (epoch+1, epochs)
	run(x_train_data, G, D, epoch, z_val_)