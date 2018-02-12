from loader import load_CIFAR10, load_data
import torch.nn as nn
from models import Model
from run import run  


nb_classes = 10
X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_CIFAR10()

X_train = X_train/255. 
X_valid = X_valid/255. 
X_test  = X_test/255.
Y_train = Y_train
Y_valid = Y_valid
Y_test  = Y_test

batch_size = 32
train_data = load_data(X_train, Y_train, batch_size)
valid_data = load_data(X_valid, Y_valid, batch_size)

model = Model(nb_classes, batch_size).cuda()

epochs = 10
for i in range(epochs):
	print 'epoch: %d/%d' % (i+1, epochs)  
	run(train_data, model, batch_size, train=True)
	run(valid_data, model, batch_size, train=False)

# pytorch
# 1 - loader do dataset
# 2 - loader para pytorch
# 3 - foward do train (perceptron)
# 4 - loss function e optimizer
# 5 - backward
# 6 - iterations e validation
# 7 - generator, discriminator 