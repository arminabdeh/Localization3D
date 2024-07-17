import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from luenn.model.model import UNet
from luenn.evaluate import reg_classification
import matplotlib.pyplot as plt
import yaml
from types import SimpleNamespace
from ruamel.yaml import YAML


def auto_scaling(param):
	bg_uniform = param.Simulation.intensity_mu_sig[0] / 100.
	bg_max = bg_uniform * 1.2
	input_offset = bg_uniform
	input_scale = param.Simulation.intensity_mu_sig[0] / 50.
	phot_max = param.Simulation.intensity_mu_sig[0] + (param.Simulation.intensity_mu_sig[1] * 8)
	z_max = param.Simulation.emitter_extent[2][1] * 1.2
	emitter_label_photon_min = param.Simulation.intensity_mu_sig[0] / 20.
	param.Simulation.bg_uniform = bg_uniform
	param.Scaling.bg_max = bg_max
	param.Scaling.input_offset = input_offset
	param.Scaling.input_scale = input_scale
	param.Scaling.phot_max = phot_max
	param.Scaling.z_max = z_max
	param.HyperParameter.emitter_label_photon_min = emitter_label_photon_min
	param.post_processing.simulation.Imean = param.Simulation.intensity_mu_sig[0]
	param.post_processing.simulation.Isig = param.Simulation.intensity_mu_sig[1]
	return param


def simple_namespace_representer(dumper, data):
	return dumper.represent_mapping(u'tag:yaml.org,2002:map', vars(data))


yaml = YAML()
yaml.representer.add_representer(SimpleNamespace, simple_namespace_representer)


def convert_to_recursive_namespace(dictionary):
	for key, value in dictionary.items():
		if isinstance(value, dict):
			dictionary[key] = convert_to_recursive_namespace(value)
	return SimpleNamespace(**dictionary)


def param_load(load_directory):
	with open(load_directory, 'r') as f:
		param_dict = yaml.load(f)
	# Convert the loaded dictionary to a RecursiveNamespace
	param_recursive_namespace = convert_to_recursive_namespace(param_dict)
	return param_recursive_namespace


def param_save(param, filename):
	if filename.endswith('.yaml'):
		with open(filename, 'w') as yaml_file:
			yaml.dump(param, yaml_file)
	else:
		raise ValueError('Filename must end with .yaml')


def generate_unique_filename(f, prefix="", extension=""):
	timestamp = datetime.now().strftime("%Y.%m.%d")
	unique_filename = f"{prefix}_{timestamp}_{str(f)}xframe{extension}"
	return unique_filename


def load_model(dir_model):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = UNet()
	model.to(device)
	loaded_model = torch.load(dir_model, map_location=device)
	if isinstance(loaded_model, dict):
		checkpoint = loaded_model['model_state_dict']
	else:
		checkpoint = loaded_model.state_dict()
	model.load_state_dict(checkpoint)
	return model


def dec_luenn_gt_transform(tar_em):
	xyz = tar_em.xyz_px.numpy().tolist()
	frame_id = tar_em.frame_ix.numpy().tolist()
	photons = tar_em.phot.numpy().tolist()
	gt = pd.DataFrame({'xyz': xyz, 'frame_id': frame_id, 'photons': photons})

	gt_list = []

	for f in range(gt['frame_id'].max() + 1):
		frame_gt = gt[gt['frame_id'] == f]

		for nn, (xyz_data, photon) in enumerate(
				zip(frame_gt['xyz'], frame_gt['photons'])):
			xf = xyz_data[1] * 4
			yf = xyz_data[0] * 4
			xi = int(xf)
			yi = int(yf)
			xm = xf - xi
			ym = yf - yi
			if xm >= 0.5:
				xi += 1
				xm -= 1
			if ym >= 0.5:
				yi += 1
				ym -= 1
			gt_frame = {
				'frame_id': f + 1,
				'seed_id': nn + 1,
				'X_tr_px': xyz_data[0],
				'Y_tr_px': xyz_data[1],
				'X_tr_nm': xyz_data[0] * 100.0,
				'Y_tr_nm': xyz_data[1] * 100.0,
				'Z_tr_nm': xyz_data[2],
				'i_tr': xi,
				'j_tr': yi,
				'photons': photon
			}
			gt_list.append(gt_frame)

	GT_Frames = pd.DataFrame(gt_list)
	return GT_Frames


def complex_real_map(x):
	xnorm, xphi, var_ch1, var_ch2 = np.array([]), np.array([]), np.array([]), np.array([])
	if isinstance(x, torch.Tensor):
		x = x.cpu().detach().numpy()
		if x.shape[-1] == 256:
			x = np.moveaxis(x, 1, -1)
	if x.ndim == 4 and x.shape[-1] == 4:
		xnorm = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
		xphi = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
		var_ch1 = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
		var_ch2 = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
		for i in range(x.shape[0]):
			xnorm[i] += (x[i, :, :, 0] ** 2 + x[i, :, :, 1] ** 2) ** .5
			xphi[i] += np.arccos(x[i, :, :, 0] / xnorm[i])
			var_ch1[i] += x[i, :, :, 2]
			var_ch2[i] += x[i, :, :, 3]
		return xnorm, xphi, var_ch1, var_ch2
	elif x.ndim == 4 and x.shape[-1] == 2:
		xnorm = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
		xphi = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
		for i in range(x.shape[0]):
			xnorm[i] += (x[i, :, :, 0] ** 2 + x[i, :, :, 1] ** 2) ** .5
			xphi[i] += np.arccos(x[i, :, :, 0] / xnorm[i])
		return xnorm, xphi, var_ch1, var_ch2
	else:
		print('Input shape not supported')


def report_performance(gt_pr):
	gt_tp = gt_pr[gt_pr['label'] == 'TP']
	gt_fp = gt_pr[gt_pr['label'] == 'FP']
	gt_fn = gt_pr[gt_pr['label'] == 'FN']
	ji = reg_classification(gt_pr).jaccardian_index()
	pr = reg_classification(gt_pr).precision()
	re = reg_classification(gt_pr).recall()
	rmse_3d = reg_classification(gt_pr).rmse_3d()
	rmse_2d = reg_classification(gt_pr).rmse_2d()
	rmse_z = reg_classification(gt_pr).rmse_z()
	rmse_x = reg_classification(gt_pr).rmse_x()
	rmse_y = reg_classification(gt_pr).rmse_y()
	del_x = reg_classification(gt_pr).del_x()
	del_y = reg_classification(gt_pr).del_y()
	del_z = reg_classification(gt_pr).del_z()
	n_tp = len(gt_tp)
	n_fp = len(gt_fp)
	n_fn = len(gt_fn)
	print(f'number of TP is {n_tp}')
	print(f'number of FP is {n_fp}')
	print(f'number of FN is {n_fn}')

	print(f'ji is {ji}')
	print(f'rmse_3d is {rmse_3d / np.sqrt(3)}')
	print(f'rmse_2d is {rmse_2d / np.sqrt(2)}')
	print(f'rmse_x is {rmse_x}')
	print(f'rmse_y is {rmse_y}')
	print(f'rmse_z is {rmse_z}')
	print(f'del_x is {del_x}')
	print(f'del_y is {del_y}')
	print(f'del_z is {del_z}')
	print('-' * 50)
	return re, pr, ji, rmse_3d, rmse_2d, rmse_z, rmse_x, rmse_y, del_x, del_y, del_z


def param_reference():
	dir = os.path.dirname(os.path.dirname(__file__))
	param_path = os.path.join(dir, 'config/param/', 'param_reference.yaml')
	param_ref = param_load(param_path)
	calib_path = os.path.join(dir, 'config/calib/', 'spline_calibration_3d_as_3dcal.mat')
	param_ref.InOut.calibration_file = calib_path
	return param_ref


def visualize_results(pr_gt, temp_out):
	frame_id = pr_gt['frame_id'].min()
	pr_gt_frame = pr_gt[pr_gt['frame_id'] == frame_id]
	pr_gt_tp = pr_gt_frame[pr_gt_frame['label'] == 'TP']
	pr_gt_fp = pr_gt_frame[pr_gt_frame['label'] == 'FP']
	pr_gt_fn = pr_gt_frame[pr_gt_frame['label'] == 'FN']



	frame = temp_out[frame_id - 1, :, :, :]
	frame_norm = np.sqrt(np.square(frame[:, :, 0]) + np.square(frame[:, :, 1]))
	frame_phi = np.arccos(frame[:, :, 0] / (frame_norm + 1e-10))
	dpi = 150
	colors = ['red', 'blue', 'violet', 'yellow']
	plt.rcParams['font.family'] = 'serif'
	figure, ax = plt.subplots(2,2, figsize=(10, 10), dpi=dpi)
	figure.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the spacing between subplots
	ax[0, 0].imshow(frame_norm, origin='lower')
	ax[0, 0].set_title(f"prediction lateral, min = {np.min(frame_norm):.2f}, max = {np.max(frame_norm):.2f}")
	ax[0, 1].imshow(frame[:, :, 0], origin='lower')
	ax[0, 1].set_title(f"channel sin, min = {np.min(frame[:, :, 0]):.2f}, max = {np.max(frame[:, :, 0]):.2f}")
	ax[1, 0].imshow(frame[:, :, 1], origin='lower')
	ax[1, 0].set_title(f"channel cos, min = {np.min(frame[:, :, 1]):.2f}, max = {np.max(frame[:, :, 1]):.2f}")
	ax[1, 1].imshow(frame_phi, origin='lower')
	ax[1, 1].set_title(f"prediction Z, min = {np.min(frame_phi):.2f}, max = {np.max(frame_phi):.2f}")
	for i in range(2):
		for j in range(2):
			ax[i, j].set_xticks([])
			ax[i, j].set_yticks([])
			if not pr_gt_tp.empty:
				ax[i, j].scatter(pr_gt_tp.X_tr_px*4., pr_gt_tp.Y_tr_px*4., color=colors[0], s=1, label='Ground Truth')
				ax[i, j].scatter(pr_gt_tp.X_pr_px*4., pr_gt_tp.Y_pr_px*4., color=colors[1], s=1, label='Prediction')
			if not pr_gt_fp.empty:
				ax[i, j].scatter(pr_gt_fp.X_pr_px*4., pr_gt_fp.Y_pr_px*4., color=colors[2], s=5, label='False Positive', marker='X', alpha=0.8)
			if not pr_gt_fn.empty:
				ax[i, j].scatter(pr_gt_fn.X_tr_px*4., pr_gt_fn.Y_tr_px*4., color=colors[3], s=5, label='False Negative', marker='s', alpha=0.8)

	return figure

def visualize_results_corr(pr_gt):

	pr_gt_scatter = pr_gt.copy()
	pr_gt_scatter = pr_gt_scatter[pr_gt_scatter['label'] == 'TP']

	if pr_gt_scatter.empty:
		pr_gt_scatter = pd.DataFrame({'X_tr_nm': [], 'Y_tr_nm': [], 'Z_tr_nm': [],
									  'X_pr_nm': [], 'Y_pr_nm': [], 'Z_pr_nm': [],
									  'X_var': [], 'Y_var': [], 'Z_var': [],'var_3d': [], 'var_2d': []})
	# errors
	pr_gt_scatter['x_err'] = abs(pr_gt_scatter.X_tr_nm - pr_gt_scatter.X_pr_nm)
	pr_gt_scatter['y_err'] = abs(pr_gt_scatter.Y_tr_nm - pr_gt_scatter.Y_pr_nm)
	pr_gt_scatter['z_err'] = abs(pr_gt_scatter.Z_tr_nm - pr_gt_scatter.Z_pr_nm)
	pr_gt_scatter['tot_err'] = np.sqrt(pr_gt_scatter['x_err'] ** 2 + pr_gt_scatter['y_err'] ** 2 + pr_gt_scatter['z_err'] ** 2)
	pr_gt_scatter['lat_err'] = np.sqrt(pr_gt_scatter['x_err'] ** 2 + pr_gt_scatter['y_err'] ** 2)
	pr_gt_scatter['X_var'] = abs(pr_gt_scatter['X_var'].values)
	pr_gt_scatter['Y_var'] = abs(pr_gt_scatter['Y_var'].values)
	pr_gt_scatter['Z_var'] = abs(pr_gt_scatter['Z_var'].values)
	# Calculate the correlation between x_err and y_err
	corr_total = pr_gt_scatter[['tot_err', 'var_3d']].corr()
	print(f"correlation total variance: \n{corr_total}")
	cor_x = pr_gt_scatter[['x_err', 'X_var']].corr()
	print(f"correlation x variance: \n{cor_x}")
	cor_y = pr_gt_scatter[['y_err', 'Y_var']].corr()
	print(f"correlation y variance: \n{cor_y}")
	cor_z = pr_gt_scatter[['z_err', 'Z_var']].corr()
	print(f"correlation z variance: \n{cor_z}")

	print('report:')
	print('\033[1m' + 'X Direction' + '\033[0m')
	print(f" ERR ==>  mean: {pr_gt_scatter['x_err'].mean()}, max: {pr_gt_scatter['x_err'].max()}, min: {pr_gt_scatter['x_err'].min()}")
	print(f" VAR ==>  mean: {pr_gt_scatter['X_var'].mean()}, max: {pr_gt_scatter['X_var'].max()}, min: {pr_gt_scatter['X_var'].min()}")
	print('\033[1m' + 'Y Direction' + '\033[0m')
	print(f" ERR ==>  mean: {pr_gt_scatter['y_err'].mean()}, max: {pr_gt_scatter['y_err'].max()}, min: {pr_gt_scatter['y_err'].min()}")
	print(f" VAR ==>  mean: {pr_gt_scatter['Y_var'].mean()}, max: {pr_gt_scatter['Y_var'].max()}, min: {pr_gt_scatter['Y_var'].min()}")
	print('\033[1m' + 'Z Direction' + '\033[0m')
	print(f" ERR ==>  mean: {pr_gt_scatter['z_err'].mean()}, max: {pr_gt_scatter['z_err'].max()}, min: {pr_gt_scatter['z_err'].min()}")
	print(f" VAR ==>  mean: {pr_gt_scatter['Z_var'].mean()}, max: {pr_gt_scatter['Z_var'].max()}, min: {pr_gt_scatter['Z_var'].min()}")
	print('\033[1m' + 'LATERAL' + '\033[0m')
	print(f" ERR ==>  mean: {pr_gt_scatter['lat_err'].mean()}, max: {pr_gt_scatter['lat_err'].max()}, min: {pr_gt_scatter['lat_err'].min()}")
	print(f" VAR ==>  mean: {pr_gt_scatter['var_2d'].mean()}, max: {pr_gt_scatter['var_2d'].max()}, min: {pr_gt_scatter['var_2d'].min()}")
	print('\033[1m' + 'TOTAL' + '\033[0m')
	print(f" ERR ==>  mean: {pr_gt_scatter['tot_err'].mean()}, max: {pr_gt_scatter['tot_err'].max()}, min: {pr_gt_scatter['tot_err'].min()}")
	print(f" VAR ==>  mean: {pr_gt_scatter['var_3d'].mean()}, max: {pr_gt_scatter['var_3d'].max()}, min: {pr_gt_scatter['var_3d'].min()}")
	print('-' * 50)

	figure, ax = plt.subplots(2, 2, figsize=(8, 8), dpi=80)
	figure.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the spacing between subplots
	titles = ['Error 3D', 'Error 2D', 'Error Z', 'Error X']
	ylabels = ['Variance 3D', 'Variance 2D', 'Variance Z', 'Variance X']
	xlabels = ['Error 3D', 'Error 2D', 'Error Z', 'Error X']

	ax[0, 0].scatter(pr_gt_scatter['tot_err'], pr_gt_scatter['var_3d'], color='black', s=4, label='3D', alpha=0.5)
	ax[0, 0].plot([0, max(pr_gt_scatter['tot_err'].max(),pr_gt_scatter['var_3d'].max())],
				  [0, max(pr_gt_scatter['tot_err'].max(),pr_gt_scatter['var_3d'].max())], color='blue', linestyle='--', alpha=0.5)

	ax[0, 1].scatter(pr_gt_scatter['lat_err'], pr_gt_scatter['var_2d'], color='blue', s=4, label='Lateral', alpha=0.5)
	ax[0, 1].plot([0, max(pr_gt_scatter['lat_err'].max(),pr_gt_scatter['var_2d'].max())],
				  [0, max(pr_gt_scatter['lat_err'].max(),pr_gt_scatter['var_2d'].max())], color='blue', linestyle='--', alpha=0.5)

	ax[1, 0].scatter(pr_gt_scatter['z_err'], pr_gt_scatter['Z_var'], color='green', s=4, label='Z direction', alpha=0.5)
	ax[1, 0].plot([0, max(pr_gt_scatter['z_err'].max(),pr_gt_scatter['Z_var'].max())],
				  [0, max(pr_gt_scatter['z_err'].max(),pr_gt_scatter['Z_var'].max())], color='blue', linestyle='--', alpha=0.5)

	ax[1, 1].scatter(abs(pr_gt_scatter['x_err']), pr_gt_scatter['X_var'], color='red', s=3, label='X direction', alpha=0.4)
	ax[1, 1].scatter(abs(pr_gt_scatter['y_err'].abs()), pr_gt_scatter['Y_var'].abs(), color='blue', s=3, label='Y direction', alpha=0.4)
	ax[1, 1].plot([0, max(pr_gt_scatter['x_err'].max(),pr_gt_scatter['X_var'].max(),pr_gt_scatter['Y_var'].max(),pr_gt_scatter['y_err'].max())],
				  [0, max(pr_gt_scatter['x_err'].max(),pr_gt_scatter['X_var'].max(),pr_gt_scatter['Y_var'].max(),pr_gt_scatter['y_err'].max())], color='blue', linestyle='--', alpha=0.5)


	for i in range(2):
		for j in range(2):
			ax[i, j].set_xlabel(xlabels[i*2+j])
			ax[i, j].set_ylabel(ylabels[i*2+j])
			ax[i, j].set_title(titles[i*2+j])
			ax[i, j].legend()
	plt.tight_layout()
	plt.close('all')
	return figure

def pre_trained_model():
	dir = os.path.dirname(os.path.dirname(__file__))
	model_path = os.path.join(dir, 'config/model/', 'model_reference.pth')
	model = load_model(model_path)
	return model


if __name__ == '__main__':
	print(param_reference())
	param_save(param_reference(), 'param_reference.yaml')
