import torch
import torch.nn as nn
from torchsummary import summary


# ,kernel_size_encoder,kernel_size_decoder,kernel_size_output

class UNet(nn.Module):
	def __init__(self, initializer, activation,
				 input_channels, pred_channels, kernel_size_encoder,
				 kernel_size_decoder, kernel_size_output):
		super(UNet, self).__init__()
		self.initializer = initializer
		self.activation = activation
		self.enc = nn.ModuleList([
			nn.Conv2d(input_channels, 64, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 1x64x64 -> 64x64x64 layer_id 0
			self.activation(inplace=True),  # layer_id 1
			nn.BatchNorm2d(64),  # layer_id 2
			nn.Conv2d(64, 64, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 64x64x64 -> 64x64x64 layer_id 3
			self.activation(inplace=True),  # layer_id 4
			nn.BatchNorm2d(64),  # layer_id 5 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2),  # 64-->32 layer_id 6
			nn.Conv2d(64, 128, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 64x32x32 -> 128x32x32 layer_id 7
			self.activation(inplace=True),  # layer_id 8
			nn.BatchNorm2d(128),  # layer_id 9
			nn.Conv2d(128, 128, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 128x32x32 -> 128x32x32 layer_id 10
			self.activation(inplace=True),  # layer_id 11
			nn.BatchNorm2d(128),  # layer_id 12 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2),  # 32-->16 layer_id 13
			nn.Conv2d(128, 256, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 128x16x16 -> 256x16x16 layer_id 14
			self.activation(inplace=True),  # layer_id 15
			nn.BatchNorm2d(256),  # layer_id 16
			nn.Conv2d(256, 256, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 256x16x16 -> 256x16x16 layer_id 17
			self.activation(inplace=True),  # layer_id 18
			nn.BatchNorm2d(256),  # layer_id 19
			nn.Conv2d(256, 256, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 256x16x16 -> 256x16x16 layer_id 20
			self.activation(inplace=True),  # layer_id 21
			nn.BatchNorm2d(256),  # layer_id 22
			nn.Conv2d(256, 256, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 256x16x16 -> 256x16x16 layer_id 23
			self.activation(inplace=True),  # layer_id 24
			nn.BatchNorm2d(256),  # layer_id 25 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2),  # 16-->8 layer_id 26
			nn.Conv2d(256, 512, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 256x8x8 -> 512x8x8 layer_id 27
			self.activation(inplace=True),  # layer_id 28
			nn.BatchNorm2d(512),  # layer_id 29
			nn.Conv2d(512, 512, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 512x8x8 -> 512x8x8 layer_id 30
			self.activation(inplace=True),  # layer_id 31
			nn.BatchNorm2d(512),  # layer_id 32
			nn.Conv2d(512, 512, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 512x8x8 -> 512x8x8 layer_id 33
			self.activation(inplace=True),  # layer_id 34
			nn.BatchNorm2d(512),  # layer_id 35
			nn.Conv2d(512, 512, kernel_size=kernel_size_encoder, stride=1, padding=kernel_size_encoder // 2),
			# 512x8x8 -> 512x8x8 layer_id 36
			self.activation(inplace=True),  # layer_id 37
			nn.BatchNorm2d(512)])  # layer_id 38

		self.decoder = nn.ModuleList([
			nn.UpsamplingBilinear2d(scale_factor=2),  # 8-->16 layer_id 0
			nn.Conv2d(512, 256, kernel_size=kernel_size_decoder, stride=1, padding=kernel_size_decoder // 2),
			# 512x16x16 -> 256x16x16 layer_id 1
			self.activation(inplace=True),  # layer_id 2
			nn.BatchNorm2d(256),  # layer_id 3 --> skip connection to layer_id 25
			nn.Conv2d(512, 256, kernel_size=kernel_size_decoder, stride=1, padding=kernel_size_decoder // 2),
			# 512x16x16 -> 256x16x16 layer_id 4
			self.activation(inplace=True),  # layer_id 5
			nn.BatchNorm2d(256),  # layer_id 6
			nn.UpsamplingBilinear2d(scale_factor=2),  # 16-->32 layer_id 7
			nn.Conv2d(256, 128, kernel_size=kernel_size_decoder, stride=1, padding=kernel_size_decoder // 2),
			# 256x32x32 -> 128x32x32 layer_id 8
			self.activation(inplace=True),  # layer_id 9
			nn.BatchNorm2d(128),  # layer_id 10 --> skip connection to layer_id 12
			nn.Conv2d(256, 128, kernel_size=kernel_size_decoder, stride=1, padding=kernel_size_decoder // 2),
			# 256x32x32 -> 128x32x32 layer_id 11
			self.activation(inplace=True),  # layer_id 12
			nn.BatchNorm2d(128),  # layer_id 13
			nn.UpsamplingBilinear2d(scale_factor=2),  # 32-->64 layer_id 14
			nn.Conv2d(128, pred_channels, kernel_size=kernel_size_decoder, stride=1, padding=1),
			# 128x64x64 -> 64x64x64 layer_id 15
			self.activation(inplace=True),  # layer_id 16
			nn.BatchNorm2d(pred_channels),  # layer_id 17 --> skip connection to layer_id 5
			nn.Conv2d(pred_channels + 64, pred_channels, kernel_size=kernel_size_decoder, stride=1,
					  padding=kernel_size_decoder // 2),  # 128x64x64 -> 64x64x64 layer_id 18
			self.activation(inplace=True),  # layer_id 19
			nn.BatchNorm2d(pred_channels),  # layer_id 20
			nn.UpsamplingBilinear2d(scale_factor=2),  # 64-->128 layer_id 21
			nn.Conv2d(pred_channels, pred_channels, kernel_size=kernel_size_decoder, stride=1,
					  padding=kernel_size_decoder // 2),  # 64x128x128 -> 64128x128 layer_id 22
			self.activation(inplace=True),  # layer_id 23
			nn.BatchNorm2d(pred_channels),  # layer_id 24
			nn.Conv2d(pred_channels, pred_channels, kernel_size=kernel_size_decoder, stride=1,
					  padding=kernel_size_decoder // 2),  # 64x128x128 -> 64x128x128 layer_id 25
			self.activation(inplace=True),  # layer_id 26
			nn.BatchNorm2d(pred_channels),  # layer_id 27 --> recurrent connection to layer_id 24
			nn.Conv2d(pred_channels * 2, pred_channels, kernel_size=kernel_size_decoder, stride=1,
					  padding=kernel_size_decoder // 2),  # 64x128x128 -> 64x128x128 layer_id 28
			self.activation(inplace=True),  # layer_id 29
			nn.BatchNorm2d(pred_channels),  # layer_id 30 --> recurrent connection to layer_id 24 and 27
			nn.Conv2d(pred_channels * 3, pred_channels, kernel_size=kernel_size_decoder, stride=1,
					  padding=kernel_size_decoder // 2),  # 64x128x128 -> 64x128x128 layer_id 31
			self.activation(inplace=True),  # layer_id 32
			nn.BatchNorm2d(pred_channels),  # layer_id 33
			nn.UpsamplingBilinear2d(scale_factor=2),  # 128-->256 layer_id 34
			nn.Conv2d(pred_channels, pred_channels, kernel_size=kernel_size_decoder, stride=1,
					  padding=kernel_size_decoder // 2),  # pred_channelsx256x256 -> pred_channelsx256x256 layer_id 35
			self.activation(inplace=True),  # layer_id 36
			nn.BatchNorm2d(pred_channels),  # layer_id 37
			nn.Conv2d(pred_channels, pred_channels, kernel_size=kernel_size_decoder, stride=1,
					  padding=kernel_size_decoder // 2),  # pred_channelsx256x256 -> pred_channelsx256x256 layer_id 38
			self.activation(inplace=True),  # layer_id 39
			nn.BatchNorm2d(pred_channels),  # layer_id 40 --> recurrent connection to layer_id 37
			nn.Conv2d(pred_channels * 2, pred_channels, kernel_size=kernel_size_decoder, stride=1,
					  padding=kernel_size_decoder // 2),  # pred_channelsx256x256 -> pred_channelsx256x256 layer_id 41
			self.activation(inplace=True),  # layer_id 42
			nn.BatchNorm2d(pred_channels),  # layer_id 43 --> recurrent connection to layer_id 37 and 40
			nn.Conv2d(pred_channels * 3, pred_channels, kernel_size=kernel_size_decoder, stride=1,
					  padding=kernel_size_decoder // 2),  # pred_channelsx256x256 -> pred_channelsx256x256 layer_id 44
			self.activation(inplace=True),  # layer_id 45
			nn.BatchNorm2d(pred_channels),  # layer_id 46 --> recurrent connection to layer_id 37, 40 and 43
			nn.Conv2d(pred_channels, 3, kernel_size=kernel_size_output, stride=1,
					  padding=kernel_size_output // 2)])  # output layer_id 47
		# initializer to initialize the weights of the model
		for layer in self.enc:
			if isinstance(layer, nn.Conv2d):
				self.initializer(layer.weight.data)
		for layer in self.decoder:
			if isinstance(layer, nn.Conv2d):
				self.initializer(layer.weight.data)

	def forward(self, x):
		enc_skip = []
		for i, layer in enumerate(self.enc.children()):
			x = layer(x)
			print(x.shape)
			if i in {5, 12, 25}:
				enc_skip.append(x)
		for i, layer in enumerate(self.decoder.children()):
			x = layer(x)
			if i in {3, 10, 17}:
				x = torch.cat((x, enc_skip.pop()), dim=1)
			if i == 24:
				x_rec1 = x
			if i == 27 or i == 30:
				x = torch.cat((x, x_rec1), dim=1)
				x_rec1 = x
			if i == 37:
				x_rec2 = x
			if i == 40 or i == 43:
				x = torch.cat((x, x_rec2), dim=1)
				x_rec2 = x
		return x


if __name__ == "__main__":
	initializer = nn.init.kaiming_normal_
	activation = nn.ELU
	input_channels = 1
	pred_channels = 64
	kernel_size_encoder = 3
	kernel_size_decoder = 3
	kernel_size_output = 3
	model = UNet(initializer, activation, input_channels, pred_channels, kernel_size_encoder, kernel_size_decoder,
				 kernel_size_output)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	print('model summary: ')
	summary(model, input_size=(1, 64, 64), batch_size=1)
