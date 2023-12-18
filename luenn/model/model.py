import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
class parallel_conv(nn.Module):
	def __init__(self,input_channels=1):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2)
		self.conv3 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3)
	def forward(self, x):
		x1 = F.elu(self.conv1(x))
		x2 = F.elu(self.conv2(x))
		x3 = F.elu(self.conv3(x))
		x = torch.cat((x1, x2, x3), dim=1)
		return x


class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		self.initializer = nn.init.xavier_uniform_
		self.activation = nn.ELU
		self.input = parallel_conv()
		self.enc = nn.ModuleList([
			nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1), # 1x64x64 -> 64x64x64 layer_id 0
			self.activation(),  # layer_id 1
			nn.BatchNorm2d(64),  # layer_id 2
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64x64x64 -> 64x64x64 layer_id 3
			self.activation(),  # layer_id 4
			nn.BatchNorm2d(64),  # layer_id 5 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2),  # 64-->32 layer_id 6
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 64x32x32 -> 128x32x32 layer_id 7
			self.activation(),  # layer_id 8
			nn.BatchNorm2d(128),  # layer_id 9
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 128x32x32 -> 128x32x32 layer_id 10
			self.activation(),  # layer_id 11
			nn.BatchNorm2d(128),  # layer_id 12 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2),  # 32-->16 layer_id 13
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 128x16x16 -> 256x16x16 layer_id 14
			self.activation(),  # layer_id 15
			nn.BatchNorm2d(256),  # layer_id 16
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 256x16x16 -> 256x16x16 layer_id 17
			self.activation(),  # layer_id 18
			nn.BatchNorm2d(256),  # layer_id 19
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 256x16x16 -> 256x16x16 layer_id 20
			self.activation(),  # layer_id 21
			nn.BatchNorm2d(256),  # layer_id 22
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 256x16x16 -> 256x16x16 layer_id 23
			self.activation(),  # layer_id 24
			nn.BatchNorm2d(256),  # layer_id 25 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2),  # 16-->8 layer_id 26
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x8x8 -> 512x8x8 layer_id 27
			self.activation(),  # layer_id 28
			nn.BatchNorm2d(512),  # layer_id 29
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512x8x8 -> 512x8x8 layer_id 30
			self.activation(),  # layer_id 31
			nn.BatchNorm2d(512),  # layer_id 32
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# 512x8x8 -> 512x8x8 layer_id 33
			self.activation(),  # layer_id 34
			nn.BatchNorm2d(512),  # layer_id 35
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# 512x8x8 -> 512x8x8 layer_id 36
			self.activation(),  # layer_id 37
			nn.BatchNorm2d(512),  # layer_id 38 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2), # 8-->4 layer_id 39
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512x4x4 -> 512x4x4 layer_id 40
			self.activation(),  # layer_id 41
			nn.BatchNorm2d(512),  # layer_id 42
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512x4x4 -> 512x4x4 layer_id 43
			self.activation(),  # layer_id 44
			nn.BatchNorm2d(512),  # layer_id 45
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512x4x4 -> 512x4x4 layer_id 46
			self.activation(),  # layer_id 47
			nn.BatchNorm2d(512),  # layer_id 48
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512x4x4 -> 512x4x4 layer_id 49
			self.activation(),  # layer_id 50
			nn.BatchNorm2d(512)]) # layer_id 51


		self.decoder = nn.ModuleList([
			nn.UpsamplingBilinear2d(scale_factor=2),  # 4-->8 layer_id 0
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),	# 512x8x8 -> 512x8x8 layer_id 1
			self.activation(),  # layer_id 2
			nn.BatchNorm2d(512),  # layer_id 3 --> skip connection to layer_id 38
			nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),	# 1024x8x8 -> 512x8x8 layer_id 4
			self.activation(),  # layer_id 5
			nn.BatchNorm2d(512),  # layer_id 6
			nn.UpsamplingBilinear2d(scale_factor=2),  # 8-->16 layer_id 7
			nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),	# 512x16x16 -> 256x16x16 layer_id 8
			self.activation(),  # layer_id 9
			nn.BatchNorm2d(256),  # layer_id 10 --> skip connection to layer_id 25
			nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), # 512x16x16 -> 256x16x16 layer_id 11
			self.activation(),  # layer_id 12
			nn.BatchNorm2d(256),  # layer_id 13
			nn.UpsamplingBilinear2d(scale_factor=2),  # 16-->32 layer_id 14
			nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), # 256x32x32 -> 128x32x32 layer_id 15
			self.activation(),  # layer_id 16
			nn.BatchNorm2d(128),  # layer_id 17 --> skip connection to layer_id 12
			nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), # 256x32x32 -> 128x32x32 layer_id 18
			self.activation(),  # layer_id 19
			nn.BatchNorm2d(128),  # layer_id 20
			nn.UpsamplingBilinear2d(scale_factor=2),  # 32-->64 layer_id 21
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # 128x64x64 -> 64x64x64 layer_id 22
			self.activation(),  # layer_id 23
			nn.BatchNorm2d(64),  # layer_id 24 --> skip connection to layer_id 5
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 128x64x64 -> 64x64x64 layer_id 25
			self.activation(),  # layer_id 26
			nn.BatchNorm2d(64),  # layer_id 27
			nn.UpsamplingBilinear2d(scale_factor=2),  # 64-->128 layer_id 28
			nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),  # 64x128x128 -> 64128x128 layer_id 29
			self.activation(),  # layer_id 30
			nn.BatchNorm2d(64),  # layer_id 31
			nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),  # 64x128x128 -> 64x128x128 layer_id 32
			self.activation(),  # layer_id 33
			nn.BatchNorm2d(64),  # layer_id 34 --> recurrent connection to layer_id 31
			nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1),  # 64x128x128 -> 64x128x128 layer_id 35
			self.activation(),  # layer_id 36
			nn.BatchNorm2d(64),  # layer_id 37 --> recurrent connection to layer_id 31 and 34
			nn.Conv2d(64 * 3, 64, kernel_size=3, stride=1, padding=1),  # 64x128x128 -> 64x128x128 layer_id 38
			self.activation(),  # layer_id 39
			nn.BatchNorm2d(64),  # layer_id 40
			nn.UpsamplingBilinear2d(scale_factor=2),  # 128-->256 layer_id 41
			nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),  # 64x256x256 -> 64x256x256 layer_id 42
			self.activation(),  # layer_id 43
			nn.BatchNorm2d(64),  # layer_id 44
			nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),  # 64x256x256 -> 64x256x256 layer_id 45
			self.activation(),  # layer_id 46
			nn.BatchNorm2d(64),  # layer_id 47 --> recurrent connection to layer_id 44
			nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1),  # 64x256x256 -> 64x256x256 layer_id 48
			self.activation(),  # layer_id 49
			nn.BatchNorm2d(64),  # layer_id 50 --> recurrent connection to layer_id 44 and 47
			nn.Conv2d(64 * 3, 256, kernel_size=3, stride=1, padding=1),  # 64x256x256 -> 64x256x256 layer_id 51
			self.activation(),  # layer_id 52
			nn.Conv2d(256, 2, kernel_size=5, stride=1,  padding=2)])  # output layer_id 53
		# initializer to initialize the weights of the model
		for layer in self.enc:
			if isinstance(layer, nn.Conv2d):
				self.initializer(layer.weight.data)
		for layer in self.decoder:
			if isinstance(layer, nn.Conv2d):
				self.initializer(layer.weight.data)

	def forward(self, x):
		x = self.input(x)
		enc_skip = []
		for i, layer in enumerate(self.enc.children()):
			x = layer(x)
			if i in {5, 12, 25, 38}:
				enc_skip.append(x)
		for i, layer in enumerate(self.decoder.children()):
			x = layer(x)
			if i in {3,10,17,24}:
				x = torch.cat((x, enc_skip.pop()), dim=1)

			rec_layers1 = [34, 37]
			if i==31:
				x_rec = x
			if i in rec_layers1:
				x = torch.cat((x, x_rec), dim=1)
				x_rec = x
			# layer_id 50 --> recurrent connection to layer_id 44 and 47
			if i == 44:
				x_rec1 = x
			rec_layers2 = [47, 50]
			if i in rec_layers2:
				x = torch.cat((x, x_rec1), dim=1)
				x_rec1 = x
		return x

if __name__ == "__main__":
	from luenn.utils import param_reference
	param = param_reference()
	model = UNet()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	print('model summary: ')
	summary(model, input_size=(1, 64, 64), batch_size=1)



