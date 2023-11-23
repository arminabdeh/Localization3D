import unittest

import torch

import luenn
from luenn import model


class TestModel(unittest.TestCase):
	def test_model(self):
		model = luenn.model.UNet()
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model.to(device)
		self.assertEqual(True, True)
