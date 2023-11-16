import decode.utils.param_io as param_io
from tqdm import tqdm
import numpy as np
import pandas as pd
from luenn.localization import localizer_machine  # replace with your actual module
from luenn.generic import fly_simulator
from torch.utils.data import Dataset
import torch
from time import sleep
import multiprocessing

class CustomDataset(Dataset):
	def __init__(self, size, x, y):
		self.size = size
		self.data_x = x
		self.data_y = y
	def __len__(self):
		return self.size
	def __getitem__(self, idx):
		sleep(5)
		return self.data_x[idx], self.data_y[idx]
def custom_collate(batch):
	x_batch = torch.stack([item[0] for item in batch])
	y_batch = torch.stack([item[1] for item in batch])
	return x_batch, y_batch
def data_gen():
	multiprocessing.freeze_support()
	print('data_gen')
	param = param_io.load_params('./param/param.yaml')
	param.HyperParameter.pseudo_ds_size = 19
	param.TestSet.test_size = 9
	x_tr, y_tr, gt_tr = fly_simulator(param, report=True).ds_train()
	x_te, y_te, gt_te = fly_simulator(param, report=True).ds_test()
	stream_train = CustomDataset(19, x_tr, y_tr)
	stream_test = CustomDataset(9, x_te, y_te)
	sleep(10)
	return stream_train, stream_test, gt_tr, gt_te


def train_val_loops(model, epochs, dataloader_train, dataloader_test, gt_val,
								 criterion, optimizer, param, full_process=False):
	for epoch in range(epochs):
		model.train()
		steps_train = len(dataloader_train)
		steps_test = len(dataloader_test)
		tqdm_enum_train = tqdm(total=steps_train, smoothing=0.)
		tqdm_enum_test = tqdm(total=steps_test, smoothing=0.)
		tr_loss = 0
		for data in dataloader_train:
			inputs, labels = data
			inputs = inputs.cuda()
			labels = labels.cuda()
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			tr_loss += loss
			loss.backward()
			optimizer.step()
			tqdm_enum_train.update(1)
			inputs.cpu()
			labels.cpu()
			torch.cuda.empty_cache()
		train_loss = tr_loss / steps_train
		# validation
		val_loss = 0
		loc_file = None
		pr_gt = pd.DataFrame([])
		model.eval()
		with torch.no_grad():
			for data in dataloader_test:
				inputs, labels = data
				outputs = model(inputs.cuda())
				loss = criterion(outputs, labels.cuda())
				val_loss += loss.item()
				tqdm_enum_test.update(1)
				inputs.cpu()
				labels.cpu()
				torch.cuda.empty_cache()
				if full_process:
					temp_out = np.moveaxis(outputs.cpu().numpy(), 1, -1)
					loc_file = temp_out if loc_file is None else np.concatenate((loc_file, temp_out), axis=0)
		val_loss /= steps_test
		if full_process:
			pr_gt = localizer_machine(param, loc_file, gt_val, save=False).localization_3D()

	return model, train_loss / steps_train, val_loss, pr_gt