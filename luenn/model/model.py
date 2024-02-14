from collections import OrderedDict

import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch.distributions as dist


class ParallelConv(nn.Module):
	def __init__(self, in_channels=1):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
		self.conv3 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)

	def forward(self, x):
		return F.elu(torch.cat((self.conv1(x), self.conv2(x), self.conv3(x)), dim=1))


class UNet(nn.Module):
	def __init__(self, num_channels_out=64, kernel_size_out=5, var_channel_out=2, combined=False, batch_norm_first=True):
		super(UNet, self).__init__()
		self.activation = nn.ELU
		self.input = ParallelConv()
		self.num_channels_out = num_channels_out
		self.combined_output = combined
		def conv_block(in_channels, out_channels, batch_norm_before=batch_norm_first):
			if batch_norm_before:
				return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), self.activation())
			else:
				return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), self.activation(), nn.BatchNorm2d(out_channels))
		self.enc = nn.ModuleList([
			conv_block(128, 64),  # 0
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
			conv_block(512, 512)  # 19
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
			conv_block(64 * 3, 256),  # 21
			conv_block(256, 128),  # 22
			conv_block(128, num_channels_out),  # 23
		])
		if self.combined_output:
			self.out_all = nn.Conv2d(num_channels_out, 2+var_channel_out, kernel_size=kernel_size_out, stride=1, padding=kernel_size_out//2)
		else:
			self.out_mean = nn.Conv2d(num_channels_out, 2, kernel_size=kernel_size_out, stride=1, padding=kernel_size_out//2)
			self.out_std  = nn.Conv2d(num_channels_out, var_channel_out, kernel_size=kernel_size_out, stride=1, padding=kernel_size_out//2)

	def forward(self, x):
		x = self.input(x)
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
				x = torch.cat((x, x_rec1), dim=1)
				x_rec1 = x
			if i == 18:
				x_rec2 = x
			if i in {19, 20}:  # recurrent connection to layer_id 19
				x = torch.cat((x, x_rec2), dim=1)
				x_rec2 = x
		if self.combined_output:
			outputs = self.out(x)
			mean_ch = outputs[:, :2, :, :]
			mean_ch = torch.clamp(mean_ch, min=-1000.0, max=1000.0)
			std_ch = F.softplus(outputs[:, 2:, :, :])
			std_ch = torch.clamp(std_ch, min=1.0)
		else:
			mean_ch = self.out_mean(x)
			mean_ch = torch.clamp(mean_ch, min=-1000.0, max=1000.0)
			std_ch = F.softplus(self.out_std(x))
			std_ch = torch.clamp(std_ch, min=1.0)
		outputs = torch.cat([mean_ch, std_ch], dim=1)
		return outputs


class SmallCNNWithUncertainty(nn.Module):
	def __init__(self, input_channels=4, input_crop_size=4):
		super(SmallCNNWithUncertainty, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
		self.relu1 = nn.ReLU()
		self.batchnorm1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.relu2 = nn.ReLU()
		self.batchnorm2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.relu3 = nn.ReLU()
		self.batchnorm3 = nn.BatchNorm2d(64)
		image_size = int((input_crop_size * 2) + 1)
		self.fc_mean = nn.Linear(64 * image_size * image_size, 3)  # Output has 3 dimensions (mean_x, mean_y, mean_z)
		self.fc_std = nn.Linear(64 * image_size * image_size, 3)  # Output has 3 dimensions (std_x, std_y, std_z)
		self.fc_weight = nn.Linear(64 * image_size * image_size, 1)  # Output has 1 dimension (weight)

	def forward(self, x):
		x = self.batchnorm1(self.relu1(self.conv1(x)))
		x = self.batchnorm2(self.relu2(self.conv2(x)))
		x = self.batchnorm3(self.relu3(self.conv3(x)))
		# x = self.relu1(self.conv1(x))
		# x = self.relu2(self.conv2(x))
		x = x.reshape(x.size(0), -1)
		# Output for mean values
		mean_output = self.fc_mean(x)
		# Output for standard deviations with Softplus activation
		std_output = F.softplus(self.fc_std(x))
		# Output for weight with Sigmoid activation
		weight_output = torch.sigmoid(self.fc_weight(x))
		# Concatenate mean and std outputs
		output = torch.cat((mean_output, std_output, weight_output), dim=1)
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
