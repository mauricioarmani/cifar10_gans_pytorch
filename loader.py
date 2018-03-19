import cPickle as pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_batch(path):
	with open(path, 'r' ) as batch:
		batch = pickle.Unpickler(batch).load()
		X = np.array(batch['data'])
		Y = np.array(batch['labels'])
		X = X.reshape(10000, 3, 32, 32).astype('float')
		return  X, Y


def load_CIFAR10():
	X_train = np.empty(shape=(30000, 3, 32, 32))
	Y_train = np.empty(shape=(30000,))

	for i in range(1, 4):
		train_path = 'dataset/data_batch_%d' %i	
		X_batch, Y_batch = load_batch(train_path)
		X_train[:len(X_batch)] = X_batch
		Y_train[:len(Y_batch)] = Y_batch

	return X_train, Y_train


class Data(Dataset):
    def __init__(self, X):
	   super(Data, self).__init__()
	   self.X = X

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)

    def __add__(self, other):
        return ConcatDataset([self, other])


def load_data(x, batch_size):
	dataset = Data(x)
	data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return data