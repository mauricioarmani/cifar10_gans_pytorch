import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.utils import save_image


def run(data, G, D, epoch, z_val_):
	flip_label   = 0.1
	smooth_label = 0.3

	optimizer_d = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
	optimizer_g = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
	loss = nn.BCEWithLogitsLoss()	

	for i, X in enumerate(data):
		k = 1
		x = Variable(X.float()).cuda()
		m = len(x)
		y_= torch.Tensor(m).uniform_(1 - smooth_label, 1)
		y = Variable(y_.float()).cuda()
		w_= torch.Tensor(m).uniform_(0, smooth_label)
		w = Variable(w_.float()).cuda()

		for j in range(k):			
			D.zero_grad()
			z1 = torch.Tensor(m, 100).normal_(0,1)
			z_d = Variable(z1.float(), volatile=True).cuda()

			if torch.rand(1)[0] > flip_label:
				loss_d_x = loss(D(x), y)
			else:
				loss_d_x = loss(D(x), w)

			z_d_2 = Variable(G(z_d).data)

			if torch.rand(1)[0] > flip_label:
				loss_d_z = loss(D(z_d_2), w)
			else:
				loss_d_z = loss(D(z_d_2), y)

			loss_d = loss_d_x + loss_d_z

			loss_d.backward()
			optimizer_d.step()

		G.zero_grad()
		z2 = torch.Tensor(m, 100).normal_(0,1)
		z_g = Variable(z2.float()).cuda()
		
		if torch.rand(1)[0] > flip_label:
			loss_g = loss(D(G(z_g)), y)
		else:
			loss_g = loss(D(G(z_g)), w)

		loss_g.backward()
		optimizer_g.step()
		if i % 50 == 0:			
			print 'loss_d: {} loss_d_real {} loss_d_fake {} - loss_g: {}'.format(loss_d.data[0], loss_d_x.data[0], loss_d_z.data[0], loss_g.data[0])

	z_val = Variable(z_val_.float()).cuda()

	G.eval()
	sample = G(z_val).data
	print "saving sample...", sample.size()
	G.train()

	filename = 'results/results-{}.jpeg'.format(epoch+1)
	save_image(sample, filename, nrow=sample.shape[0]/8, normalize=True)