
import torch

def train_one_epoch(model,optimizer,criterion,dataset):
	model.train()
	train_loss = 0
	for data in dataset:
		inputs, labels = data
		inputs = inputs.cuda()
		labels = labels.cuda()
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		train_loss+=loss.item()
		loss.backward()
		optimizer.step()
		inputs.cpu()
		labels.cpu()
		torch.cuda.empty_cache()
	train_loss /= len(dataset)
	return train_loss
import pandas as pd
import numpy as np
from tqdm import tqdm
from luenn.localization import localizer_machine
def validate(param,model,criterion,dataset,gt,full_process=False):
	model.eval()
	steps = len(dataset)
	tqdm_enum = tqdm(total=steps, smoothing=0.)
	val_loss = 0
	loc_file = None
	pr_gt = pd.DataFrame([])
	with torch.no_grad():
		for data in dataset:
			inputs, labels = data
			outputs = model(inputs.cuda())
			loss = criterion(outputs, labels.cuda())
			val_loss += loss.item()
			tqdm_enum.update(1)
			inputs.cpu()
			labels.cpu()
			torch.cuda.empty_cache()
			if full_process:
				temp_out = np.moveaxis(outputs.cpu().numpy(), 1, -1)
				loc_file = temp_out if loc_file is None else np.concatenate((loc_file, temp_out), axis=0)
	val_loss /=len(dataset)
	if full_process:
		pr_gt = localizer_machine(param,loc_file,gt,save=False).localization_3D()
	return val_loss,pr_gt

if __name__ == "__main__":
	from luenn.utils import param_load,load_model
	import os
	from luenn.generic.simulator import validation_simulator
	import torch
	dir = os.path.dirname(os.path.dirname(__file__))
	param_path = os.path.join(dir,'config/param/', 'param_reference.yaml')
	param_ref = param_load(param_path)
	param_ref.TestSet.test_size = 0
	param_ref.HyperParameter.pseudo_ds_size = 0
	param_ref.post_processing.simulation.validation_size = 2000
	param_ref.post_processing.simulation.n_min = 2
	param_ref.post_processing.simulation.n_max = 2
	param_ref.HyperParameter.batch_size = 16
	param_ref.post_processing.localization.save = False
	calib_path = os.path.join(dir,'config/calib/', 'spline_calibration_3d_as_3dcal.mat')
	param_ref.InOut.calibration_file = calib_path
	param_ref.post_processing.localization.threshold_clean = 0.0004744
	param_ref.post_processing.localization.threshold_abs = 0.0005
	param_ref.post_processing.localization.threshold_distance = 4
	param_ref.post_processing.localization.threshold_freq_sum = 4.455
	param_ref.post_processing.localization.threshold_freq_max = .395
	param_ref.post_processing.localization.radius_lat = 2
	param_ref.post_processing.localization.radius_axi = 1
	param_ref.post_processing.localization.epsilon = .00005168

	model_path = os.path.join(dir,'config/model/', 'model_reference.pth')
	model = load_model(model_path,param_ref)

	x,y,gt = validation_simulator(param_ref, report=True,with_label=True).sample()
	criterion = torch.nn.MSELoss()
	from torch.utils.data import TensorDataset, DataLoader
	dataset = DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=4, shuffle=False)
	val_loss, pr_gt = validate(param_ref,model,criterion,dataset,gt,full_process=True)
	print(f'val_loss is {val_loss}')
	print('pr_gt is:')
	print(pr_gt)
	from luenn.evaluate import reg_classification
	ji = reg_classification(pr_gt).jaccardian_index()
	rmse_3d = reg_classification(pr_gt).rmse_3d()
	rmse_2d = reg_classification(pr_gt).rmse_2d()
	print(f'ji is {ji}')
	print(f'rmse_3d is {rmse_3d/np.sqrt(3)}')
	print(f'rmse_2d is {rmse_2d/np.sqrt(2)}')
