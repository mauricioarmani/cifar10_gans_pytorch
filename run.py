import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.utils import make_grid, save_image


def run(data, G, D):
	
	optimizer_d = Adam(D.parameters(), lr=1e-4)
	optimizer_g = Adam(G.parameters(), lr=1e-4)

	for i, X in enumerate(data):
		k = 1
		x_ = X[0]
		x = Variable(x_.float()).cuda()
		m = len(x)
		y_ = torch.ones(m)
		y = Variable(y_.float()).cuda()
		w_ = torch.zeros(m)
		w = Variable(w_.float()).cuda()

		loss = nn.BCEWithLogitsLoss()
		for j in range(k):			
			z1 = torch.Tensor(m, 100).uniform_(0,1)
			z_d = Variable(z1.float()).cuda()

			loss_d_x = loss(D(x), y)
			loss_d_z = loss(D(G(z_d)), w)

			loss_d = loss_d_x + loss_d_z

			loss_d.backward()
			optimizer_d.step()
			D.zero_grad()

		z2 = torch.Tensor(m, 100).uniform_(0,1)
		z_g = Variable(z2.float(), volatile=True).cuda()
		fake = G(z_g).data

		if i % 100 == 0:
			filename = '/home/mauricio/cifar10_gan/results/results-{}.jpeg'.format(i)
			save_image(fake, filename, nrow=8, normalize=True)

		loss_g = loss(D(Variable(fake)), y)
		
		loss_g.backward(retain_graph=True)
		optimizer_g.step()
		G.zero_grad()

		if i % 50 == 0:
			sig = nn.Sigmoid()
			print 'loss_d: {} - loss_g: {} - D(x): {}'.format(loss_d.data[0], loss_g.data[0], sig(D(x)).data[0])