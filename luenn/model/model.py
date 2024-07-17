from collections import OrderedDict

import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch.distributions as dist


class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		self.activation = nn.ELU
		def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
			return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
								 self.activation(), nn.BatchNorm2d(out_channels))

		self.enc = nn.ModuleList([
			conv_block(1, 64),  # 0
			conv_block(64, 64),  # 1
			nn.MaxPool2d(2, 2),  # 2
			conv_block(64, 128),  # 3
			conv_block(128, 128),  # 4
			nn.MaxPool2d(2, 2),  # 5
			conv_block(128, 256),  # 6
			conv_block(256, 256),  # 7
			conv_block(256, 256),  # 8
			conv_block(256, 256),  # 9
			nn.MaxPool2d(2, 2),  # 10
			conv_block(256, 512),  # 11
			conv_block(512, 512),  # 12
			conv_block(512, 512),  # 13
			conv_block(512, 512),  # 14
			nn.MaxPool2d(2, 2),  # 15
			conv_block(512, 512),  # 16
			conv_block(512, 512),  # 17
			conv_block(512, 512),  # 18
			conv_block(512, 512),  # 19
		])
		self.decoder = nn.ModuleList([
			nn.UpsamplingBilinear2d(scale_factor=2),  # 0
			conv_block(512+512, 512),  # 1
			conv_block(512, 512),  # 2
			nn.UpsamplingBilinear2d(scale_factor=2),  # 3
			conv_block(512+256, 256),  # 4
			conv_block(256, 256),  # 5
			nn.UpsamplingBilinear2d(scale_factor=2),  # 6
			conv_block(256+128, 128),  # 7
			conv_block(128, 128),  # 8
			nn.UpsamplingBilinear2d(scale_factor=2),  # 9
			conv_block(128+64, 64),  # 10
			conv_block(64, 64),  # 11
			nn.UpsamplingBilinear2d(scale_factor=2),  # 12
			conv_block(64, 64),  # 13
			conv_block(64, 64),  # 14
			conv_block(64 * 2, 64),  # 15
			conv_block(64 * 3, 64),  # 16
			nn.UpsamplingBilinear2d(scale_factor=2),  # 17
			conv_block(64, 64),  # 18
			conv_block(64, 64),  # 19
			conv_block(64 * 2, 64),  # 20
			conv_block(64 * 3, 64),  # 21
		])
		self.out_mean  = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		# x = self.input(x)
		enc_skip = []
		for i, layer in enumerate(self.enc.children()):
			x = layer(x)
			if i in {1, 4, 9, 14}:
				enc_skip.append(x)
		for i, layer in enumerate(self.decoder.children()):
			x = layer(x)
			if i in {0, 3, 6, 9}:
				x = torch.cat((x, enc_skip.pop()), dim=1)
			if i == 13:
				x_rec1 = x
			if i in {14, 15}:  # recurrent connection to layer_id 14
				x_rec1 = torch.cat((x, x_rec1), dim=1)
				x = x_rec1
			if i == 18:
				x_rec2 = x
			if i in {19, 20}:
				x_rec2 = torch.cat((x, x_rec2), dim=1)
				x = x_rec2
		output = self.out_mean(x)
		return output


if __name__ == "__main__":
	from luenn.utils import param_reference
	# add graph to tensorboard
	from torch.utils.tensorboard import SummaryWriter
	param = param_reference()
	model = UNet()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	print('model summary: ')
	summary(model, input_size=(1, 64, 64), batch_size=-1)
