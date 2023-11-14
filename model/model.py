import torch.nn as nn
import torch
from torchsummary import summary
# I want to modify this model in order tonn.ReLU(inplace=True) function and number of channels as parameters
# I want to add a decoder to this model

class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		#nn.ReLU(inplace=True) function is not a parameter?
		# why it only uses one time?
		self.initializer = nn.init.kaiming_normal_

		self.enc = nn.ModuleList([
			nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # 1x64x64 -> 64x64x64 layer_id 0
			nn.ReLU(inplace=True), #layer_id 1
			nn.BatchNorm2d(64), #layer_id 2
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64x64x64 -> 64x64x64 layer_id 3
			nn.ReLU(inplace=True), #layer_id 4
			nn.BatchNorm2d(64), #layer_id 5 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2), # 64-->32 layer_id 6
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #64x32x32 -> 128x32x32 layer_id 7
			nn.ReLU(inplace=True), #layer_id 8
			nn.BatchNorm2d(128), #layer_id 9
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 128x32x32 -> 128x32x32 layer_id 10
			nn.ReLU(inplace=True), #layer_id 11
			nn.BatchNorm2d(128), #layer_id 12 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2), # 32-->16 layer_id 13
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 128x16x16 -> 256x16x16 layer_id 14
			nn.ReLU(inplace=True), #layer_id 15
			nn.BatchNorm2d(256), #layer_id 16
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 256x16x16 -> 256x16x16 layer_id 17
			nn.ReLU(inplace=True), #layer_id 18
			nn.BatchNorm2d(256), #layer_id 19
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 256x16x16 -> 256x16x16 layer_id 20
			nn.ReLU(inplace=True), #layer_id 21
			nn.BatchNorm2d(256), #layer_id 22
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 256x16x16 -> 256x16x16 layer_id 23
			nn.ReLU(inplace=True), #layer_id 24
			nn.BatchNorm2d(256), #layer_id 25 --> skip connection
			nn.MaxPool2d(kernel_size=2, stride=2), # 16-->8 layer_id 26
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 256x8x8 -> 512x8x8 layer_id 27
			nn.ReLU(inplace=True), #layer_id 28
			nn.BatchNorm2d(512), #layer_id 29
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512x8x8 -> 512x8x8 layer_id 30
			nn.ReLU(inplace=True), #layer_id 31
			nn.BatchNorm2d(512), #layer_id 32
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512x8x8 -> 512x8x8 layer_id 33
			nn.ReLU(inplace=True), #layer_id 34
			nn.BatchNorm2d(512), #layer_id 35
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512x8x8 -> 512x8x8 layer_id 36
			nn.ReLU(inplace=True), #layer_id 37
			nn.BatchNorm2d(512)]) #layer_id 38

		self.decoder = nn.ModuleList([
			nn.UpsamplingBilinear2d(scale_factor=2), # 8-->16 layer_id 0
			nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), # 512x16x16 -> 256x16x16 layer_id 1
			nn.ReLU(inplace=True), #layer_id 2
			nn.BatchNorm2d(256), #layer_id 3 --> skip connection to layer_id 25
			nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # 512x16x16 -> 256x16x16 layer_id 4
			nn.ReLU(inplace=True),  # layer_id 5
			nn.BatchNorm2d(256),  # layer_id 6
			nn.UpsamplingBilinear2d(scale_factor=2), # 16-->32 layer_id 7
			nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 256x32x32 -> 128x32x32 layer_id 8
			nn.ReLU(inplace=True),  # layer_id 9
			nn.BatchNorm2d(128),  # layer_id 10 --> skip connection to layer_id 12
			nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), # 256x32x32 -> 128x32x32 layer_id 11
			nn.ReLU(inplace=True), #layer_id 12
			nn.BatchNorm2d(128), #layer_id 13
			nn.UpsamplingBilinear2d(scale_factor=2), # 32-->64 layer_id 14
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # 128x64x64 -> 64x64x64 layer_id 15
			nn.ReLU(inplace=True), #layer_id 16
			nn.BatchNorm2d(64), #layer_id 17 --> skip connection to layer_id 5
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # 128x64x64 -> 64x64x64 layer_id 18
			nn.ReLU(inplace=True), #layer_id 19
			nn.BatchNorm2d(64), #layer_id 20
			nn.UpsamplingBilinear2d(scale_factor=2), # 64-->128 layer_id 21
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64x128x128 -> 64128x128 layer_id 22
			nn.ReLU(inplace=True), #layer_id 23
			nn.BatchNorm2d(64), #layer_id 24
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64x128x128 -> 64x128x128 layer_id 25
			nn.ReLU(inplace=True), #layer_id 26
			nn.BatchNorm2d(64), #layer_id 27 --> recurrent connection to layer_id 24
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # 64x128x128 -> 64x128x128 layer_id 28
			nn.ReLU(inplace=True), #layer_id 29
			nn.BatchNorm2d(64), #layer_id 30 --> recurrent connection to layer_id 24 and 27
			nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1), # 64x128x128 -> 64x128x128 layer_id 31
			nn.ReLU(inplace=True), #layer_id 32
			nn.BatchNorm2d(64), #layer_id 33
			nn.UpsamplingBilinear2d(scale_factor=2), # 128-->256 layer_id 34
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64x256x256 -> 64x256x256 layer_id 35
			nn.ReLU(inplace=True), #layer_id 36
			nn.BatchNorm2d(64), #layer_id 37
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64x256x256 -> 64x256x256 layer_id 38
			nn.ReLU(inplace=True), #layer_id 39
			nn.BatchNorm2d(64), #layer_id 40 --> recurrent connection to layer_id 37
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # 64x256x256 -> 64x256x256 layer_id 41
			nn.ReLU(inplace=True), #layer_id 42
			nn.BatchNorm2d(64), #layer_id 43 --> recurrent connection to layer_id 37 and 40
			nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1), # 64x256x256 -> 64x256x256 layer_id 44
			nn.ReLU(inplace=True), #layer_id 45
			nn.BatchNorm2d(64), #layer_id 46 --> recurrent connection to layer_id 37, 40 and 43
			nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)]) # output layer_id 47
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
			print(i)
			print(layer)
			x = layer(x)
			if i in {5,12,25}:
				enc_skip.append(x)
		for i, layer in enumerate(self.decoder.children()):
			x = layer(x)
			if i in {3,10,17}:
				x = torch.cat((x, enc_skip.pop()), dim=1)
			if i==24:
				x_rec1 = x
			if i==27 or i==30:
				x = torch.cat((x, x_rec1), dim=1)
				x_rec1 = x
			if i==37:
				x_rec2 = x
			if i==40 or i==43:
				x = torch.cat((x, x_rec2), dim=1)
				x_rec2 = x
		return x

if __name__ == "__main__":
	model = UNet()
	model.to('cuda')
	print('model summary: ')
	summary(model, input_size=(1, 64, 64),batch_size=1)

