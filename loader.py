import cPickle as pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_batch(path):
	with open(path, 'r' ) as batch:
		batch = pickle.Unpickler(batch).load()
		X = np.array(batch['data'])
		Y = np.array(batch['labels'])
		X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('float')
		return  X, Y


def load_CIFAR10():
	X_train = np.empty(shape=(30000, 32, 32, 3))
	Y_train = np.empty(shape=(30000,))

	for i in range(1, 4):
		train_path = 'dataset/data_batch_%d' %i	
		X_batch, Y_batch = load_batch(train_path)
		X_train[:len(X_batch)] = X_batch
		Y_train[:len(Y_batch)] = Y_batch

	valid_path = 'dataset/data_batch_%d' %4
	X_valid, Y_valid = load_batch(valid_path)

	test_path = 'dataset/data_batch_%d' %5
	X_test, Y_test = load_batch(test_path)

	return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


class Data(Dataset):
    def __init__(self, X, Y):
	   super(Data, self).__init__()
	   self.X = X
	   self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

    def __add__(self, other):
        return ConcatDataset([self, other])


def load_data(x, y, batch_size):
	dataset = Data(x, y)
	data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return data