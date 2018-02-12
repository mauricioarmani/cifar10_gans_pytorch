import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.optim import Adam 


def run(data, model, batch_size, train=True):	
	loss_fn = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=1e-3)
	val_acc = 0
	for i, (x_, y_) in enumerate(data):
		model.zero_grad()
		x = Variable(x_.float()).cuda()
		y = Variable(y_.long()).cuda()

		y_pred = model(x)
		loss = loss_fn(y_pred, y)

		if train:
			loss.backward()
			optimizer.step()
			if i % 10 == 0:
				print 'train_loss: ',  loss.data[0]
		else:
			_, indices = torch.max(y_pred.data, 1)
			batch_acc = torch.eq(y.data, indices).sum()
			val_acc += batch_acc
	if not train:
		print 'valid_acc: ', float(val_acc)/(data.__len__()*batch_size) 
