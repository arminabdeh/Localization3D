import torch
from torch.utils.data import Dataset
from luenn.model import UNet
import decode.utils.param_io as param_io
from tqdm import tqdm
import numpy as np
import pandas as pd
from luenn.localization import localizer_machine  # replace with your actual module
from luenn.generic import fly_simulator
from torch.utils.data import Dataset, TensorDataset,DataLoader
import torch
def train_val_loops(model, epochs, dataloader_train, dataloader_test,optimizer):
	for epoch in range(epochs):
		model.train()
		steps_train = len(dataloader_train)
		tqdm_enum_train = tqdm(total=steps_train, smoothing=0.)
		tr_loss = 0
		criterion = torch.nn.MSELoss()
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
	# validation
	val_loss = 0
	steps_test = len(dataloader_test)
	tqdm_enum_test = tqdm(total=steps_test, smoothing=0.)
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
	val_loss = val_loss / steps_test
	return val_loss
class training_stream(Dataset):
	def __init__(self, param,simulator):
		self.data = simulator.ds_train()
		self.num_frames = self.data[0].numpy().shape[0]
	def __len__(self):
		return self.num_frames

	def __getitem__(self, index):
		x_sim, y_sim, gt_sim = self.data
		return x_sim[index], y_sim[index]

def objective(trial):
	param = param_io.load_params('./param/param.yaml')
	param.HyperParameter.pseudo_ds_size = 19
	param.TestSet.test_size = 9
	# Generate the model.
	model = UNet()
	# Generate the optimizers.
	optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
	lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
	optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
	# Generate the dataset.
	dataset = training_stream(param, simulator)
	# Generate the dataloaders.
	batch_size = trial.suggest_int("batch_size", 1, 4, log=True)
	simulator = fly_simulator(param,report=True)
	x_test, y_test, gt_test = simulator.ds_test()
	dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
	streaming_dataset = training_stream(param, simulator)
	dataloader_test  = DataLoader(dataset_test, batch_size=4,num_workers=0, shuffle=False, pin_memory=True)
	dataloader_train = DataLoader(streaming_dataset, batch_size=batch_size,num_workers=0, shuffle=False, pin_memory=True)
	dataloader = DataLoader(dataset, batch_size=batch_size)
	val_loss = train_val_loops(model, 10, dataloader_train, dataloader_test, optimizer)
	# Training of the model.
	trial.report(val_loss, epoch)
	# Handle pruning based on the intermediate value.
	if trial.should_prune():
		raise optuna.exceptions.TrialPruned()
	return val_loss
import optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=2, timeout=600)
print(study.best_trial)

# if __name__ == '__main__':
# 	param = param_io.load_params('./param/param.yaml')
# 	param.HyperParameter.pseudo_ds_size = 19
# 	param.TestSet.test_size = 9
#
# 	batch_size = 2
# 	simulator = fly_simulator(param,report=True)
# 	x_test, y_test, gt_test = simulator.ds_test()
# 	dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
# 	dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,pin_memory=True)
# 	streaming_dataset = training_stream(param, simulator)
# 	dataloader_train = DataLoader(streaming_dataset, batch_size=batch_size,num_workers=0, shuffle=True, pin_memory=True)
#
# 	model = UNet()
# 	model.to('cuda')
# 	criterion = torch.nn.MSELoss()
# 	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 	your_full_process = False
# 	for epoch in range(4):
# 		model, train_loss, val_loss, pr_gt = train_and_validate_one_epoch(model,dataloader_train,dataloader_test,gt_test,criterion,optimizer,param)
# 		print(f'Epoch {epoch} train loss is {train_loss}')
#
#
# # using optuna to optimize hyperparameters
# import optuna
# from optuna.trial import TrialState
# from optuna.study import StudyDirection