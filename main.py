from loader import load_CIFAR10, load_data
from models import Discriminator, Generator
from run import run


z_dim = 100

X_train, Y_train = load_CIFAR10()

X_train = X_train/127.5 - 1

batch_size = 32
x_train_data = load_data(X_train, Y_train, batch_size)

model_g = Generator(z_dim).cuda()
model_d = Discriminator().cuda()

epochs = 100
for i in range(epochs):
	print 'epoch: %d/%d' % (i+1, epochs)
	run(x_train_data, model_g, model_d)