import torch.nn as nn
import torch
from torchsummary import summary

class conv_block(nn.Module):
	def __init__(self, input_channels, output_channels, activation=nn.ELU(), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
		super(conv_block, self).__init__()
		self.conv = nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, stride=stride, padding=padding)
		self.act = activation
		self.bn = nn.BatchNorm2d(output_channels)

	def forward(self, x):
		out = self.conv(x)
		out = self.act(out)
		out = self.bn(out)
		return out
channels = [1,64,64,128,128,256,256,256,256,512,512,512,512]
class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		self.encoder = nn.Sequential(
		# encoder
		self.input  = conv_block(1,64)                          #0 -->64
		self.layer1 = conv_block(64,64) )                        #1 -->64
		self.max1   = nn.MaxPool2d(kernel_size = 2, stride = 2) #2 64-->32
		self.layer2 = conv_block(64,128)                        #3 -->32
		self.layer3 = conv_block(128,128)                       #4 -->32
		self.max2   = nn.MaxPool2d(kernel_size = 2, stride = 2) #5 32-->16
		self.layer4 = conv_block(128,256)                       #6 -->16
		self.layer5 = conv_block(256,256)                       #7 -->16
		self.layer6 = conv_block(256,256)                       #8 -->16
		self.layer7 = conv_block(256,256)                       #9 -->16
		self.max3   = nn.MaxPool2d(kernel_size = 2, stride = 2) #10 16-->8
		self.layer8 = conv_block(256,512)                       #11 -->8
		self.layer9 = conv_block(512,512)                       #12 -->8
		self.layer10 = conv_block(512,512)                      #13 -->8
		self.layer11 = conv_block(512,512)                      #14 -->8
		self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)#15 8-->16
		self.layer12 = conv_block(512,256)                      #16 +9 (self.layer7)
		self.layer13 = conv_block(256+256,128)                  #17
		self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)#18 16-->32
		self.layer14 = conv_block(128,128)                      #19 +4 (self.layer3)
		self.layer15 = conv_block(128+128,64)                   #20
		self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)#21 32-->64
		self.layer16 = conv_block(64, 64)                       #22 +1 (self.layer1)
		self.layer17 = conv_block(64+64, 64)                    #23
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=2)#24 64-->128
		self.layer18 = conv_block(64, 64)                       #25 + 24
		self.layer19 = conv_block(64, 64)                       #26 + 25 + 24
		self.layer20 = conv_block(64+64, 64)                    #27 + 26 + 25 + 24
		self.layer21 = conv_block(64+64+64, 64)                 #28 + 27 + 26 + 25 + 24
		self.upsample5 = nn.UpsamplingBilinear2d(scale_factor=2)#29 128-->256
		self.layer22 = conv_block(64, 64)                       #30 + 29
		self.layer23 = conv_block(64, 64)                       #31 + 30 + 29
		self.layer24 = conv_block(64+64, 64)                    #32 + 31 + 30 + 29
		self.layer25 = conv_block(64+64+64, 64)                 #33 + 32 + 31 + 30 + 29
		self.output = torch.nn.Conv2d(64, 3, kernel_size=(5, 5),stride=(1, 1), padding=(2, 2))

	def forward(self, x):
		enc_skip = []
		x = self.encoder(x)
		# x = self.layer1(x) 
		# enc_skip.append(x)
		# x = self.max1(x)
		# x = self.layer2(x)
		# x = self.layer3(x) 
		# enc_skip.append(x)
		# x = self.max2(x)
		# x = self.layer4(x)
		# x = self.layer5(x)
		# x = self.layer6(x)
		# x = self.layer7(x) 
		# enc_skip.append(x)
		# x = self.max3(x)
		# x = self.layer8(x)
		# x = self.layer9(x)
		# x = self.layer10(x)
		# x = self.layer11(x) 
		# x = self.upsample1(x)
		# x = self.layer12(x)
		# x_skip = torch.cat((x, enc_skip.pop()), dim=1)
		# x = self.layer13(x_skip)
		# x = self.upsample2(x)
		# x = self.layer14(x)
		# x_skip = torch.cat((x, enc_skip.pop()), dim=1)
		# x = self.layer15(x_skip)
		# x = self.upsample3(x)
		# x = self.layer16(x)
		# x_skip = torch.cat((x, enc_skip.pop()), dim=1)
		# x = self.layer17(x_skip)

		# x = self.upsample4(x)
		# x = self.layer18(x)
		# xx = self.layer19(x)
		# x = torch.cat((x, xx), dim=1)
		# xx = self.layer20(x)
		# x = torch.cat((x, xx), dim=1)
		# x = self.layer21(x)

		# x = self.upsample5(x)
		# x = self.layer22(x)
		# xx = self.layer23(x)
		# x = torch.cat((x, xx), dim=1)
		# xx = self.layer24(x)
		# x = torch.cat((x, xx), dim=1)
		# x = self.layer25(x)
		out = self.output(x)
		return out
	
if __name__ == "__main__":
	model = UNet()
	model.to('cuda')
	print('model summary: ')
	summary(model, input_size=(1, 64, 64))

