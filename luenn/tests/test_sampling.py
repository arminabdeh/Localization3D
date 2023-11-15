import unittest

class TestCase(unittest.TestCase):
	def test_generic(self):
		import decode.utils.param_io as param_io
		param = param_io.load_params('../../param/param.yaml')
		param.HyperParameter.pseudo_ds_size = 2
		size_img = param.HyperParameter.pseudo_ds_size+1
		from luenn.generic import fly_simulator
		x,y,gt = fly_simulator(param,report=True).ds_train()
		import torch
		assert x.shape == (size_img,1,64,64)
		assert y.shape == (size_img,3,256,256)
		assert type(x) == torch.Tensor
		assert type(y) == torch.Tensor
		assert x.device.type == 'cpu'
		assert y.device.type == 'cpu'
		print('Test Passed')
