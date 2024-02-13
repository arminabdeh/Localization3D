import os
import numpy as np
import pandas as pd
from scipy.ndimage import label as nd_label
from skimage.measure import regionprops, centroid
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
import scipy.stats as stats

from luenn.utils import generate_unique_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.seterr(divide='ignore')
np.seterr(invalid='ignore')


class localizer_machine:
	def __init__(self, psfs, gt=None, save=None, param=None):
		if isinstance(psfs, list):
			psfs = np.array(psfs)
		if torch.is_tensor(psfs):
			psfs = np.moveaxis(psfs.cpu().numpy(), 1, -1)
		psfs = psfs.astype(np.float64)
		self.psfs = psfs
		self.gt = gt
		self.gt['frame_id'].astype('int')
		self.gt['seed_id'].astype('int')
		self.param = param
		self.skip = self.param.post_processing.localization.skip
		self.norm_peak = self.param.Simulation.scale_factor
		self.z_range = self.param.Simulation.z_range
		self.threshold_freq_sum = self.param.post_processing.localization.threshold_freq_sum
		self.threshold_freq_max = self.param.post_processing.localization.threshold_freq_max
		self.threshold_clean = param.post_processing.localization.threshold_clean
		self.threshold_abs = param.post_processing.localization.threshold_abs
		self.threshold_distance = param.post_processing.localization.threshold_distance
		self.radius_lat = param.post_processing.localization.radius_lat
		self.radius_axi = param.post_processing.localization.radius_axi
		self.eps = param.post_processing.localization.epsilon
		if self.skip:
			self.i_skip_min = int(4 * param.post_processing.domain_pool[0][0])
			self.i_skip_max = int(4 * param.post_processing.domain_pool[0][1])
			self.j_skip_min = int(4 * param.post_processing.domain_pool[1][0])
			self.j_skip_max = int(4 * param.post_processing.domain_pool[1][1])
		else:
			self.i_skip_min = max(self.radius_lat, self.radius_axi)
			self.i_skip_max = 255 - max(self.radius_lat, self.radius_axi)
			self.j_skip_min = max(self.radius_lat, self.radius_axi)
			self.j_skip_max = 255 - max(self.radius_lat, self.radius_axi)
		self.x_px_size = param.Camera.px_size[0]
		self.y_px_size = param.Camera.px_size[1]
		if save is None:
			if self.param.post_processing.localization.save is None:
				self.save = False
			else:
				self.save = self.param.post_processing.localization.save
		else:
			self.save = save

	@staticmethod
	def centroid(x, w, px_size):
		w1, w2, w3 = w
		x1, x2, x3 = x

		t1 = w1 ** (x3 ** 2 - x2 ** 2)
		t2 = w2 ** (x1 ** 2 - x3 ** 2)
		t3 = w3 ** (x2 ** 2 - x1 ** 2)
		t4 = w1 ** (x3 - x2)
		t5 = w2 ** (x1 - x3)
		t6 = w3 ** (x2 - x1)

		mu = 0.5 * np.log2(t1 * t2 * t3) / np.log2(t4 * t5 * t6)

		return mu

	@staticmethod
	def match_finder(pr_frame, gt_frame):
		if pr_frame.empty:
			gt_id = gt_frame.seed_id.to_list()
			Condition = ['FN'] * len(gt_id)
			pr_id = ['NA'] * len(gt_id)
			Res_data = pd.DataFrame({'Condition': Condition, 'gt_id': gt_id, 'pr_id': pr_id})
			return Res_data
		elif gt_frame.empty:
			pr_id = list(np.arange(len(pr_frame)))
			Condition = ['FP'] * len(pr_id)
			gt_id = ['NA'] * len(pr_id)
			Res_data = pd.DataFrame({'Condition': Condition, 'gt_id': gt_id, 'pr_id': pr_id})
			return Res_data
		else:
			pred_xyz = pr_frame[['X_pr_nm', 'Y_pr_nm', 'Z_pr_nm']].to_numpy()
			true_xyz = gt_frame[['X_tr_nm', 'Y_tr_nm', 'Z_tr_nm']].to_numpy()
			# Compute 2D and 3D distances
			cost2D = cdist(true_xyz[:, :2], pred_xyz[:, :2], 'euclidean')
			cost3D = cdist(true_xyz, pred_xyz, 'euclidean')
			# Find matching pairs based on 3D distances
			dist_matrix = np.sqrt(np.sqrt(cost3D))
			tr_id, pr_id = linear_sum_assignment(dist_matrix)

			# Initialize results list
			Results = []

			# Match TP, FN, and FP
			paired_tr = []
			paired_pr = []
			for i in range(len(tr_id)):
				if cost2D[tr_id[i], pr_id[i]] <= 250. and abs(true_xyz[tr_id[i], 2] - pred_xyz[pr_id[i], 2]) <= 500.:
					ids = {'Condition': 'TP', 'gt_id': tr_id[i], 'pr_id': pr_id[i]}
					paired_tr.append(tr_id[i])
					paired_pr.append(pr_id[i])
					Results.append(ids)

			all_tr_ids = list(np.arange(len(true_xyz)))
			T_Fals_id = list(set(all_tr_ids) - set(paired_tr))

			all_pr_ids = list(np.arange(len(pred_xyz)))
			P_Fals_id = list(set(all_pr_ids) - set(paired_pr))

			T_Fals_id.sort()
			for tt in T_Fals_id:
				ids = {'Condition': 'FN', 'gt_id': tt, 'pr_id': 'NA'}
				Results.append(ids)

			P_Fals_id.sort()
			for pp in P_Fals_id:
				ids = {'Condition': 'FP', 'gt_id': 'NA', 'pr_id': pp}
				Results.append(ids)

			# Convert results to DataFrame
			Res_data = pd.DataFrame(Results)
			return pd.DataFrame(Results)

	def var_analysis(self, variance):
		var_ch1 = variance[:, :, 0]
		var_ch2 = variance[:, :, 1]
		var_norm = (var_ch1 ** 2 + var_ch2 ** 2) ** .5
		var_z = var_ch1 / var_norm
		var_total = var_norm + var_z
		std_ch1 = np.sqrt(var_ch1)
		std_ch2 = np.sqrt(var_ch2)
		std_total = np.sqrt(var_total)
		return var_total, std_total, std_ch1, std_ch2, var_ch1, var_ch2

	def seed_candidate_3D(self, psf):
		try:
			psfs_cos = psf[:, :, 0] / self.norm_peak
			psfs_sin = psf[:, :, 1] / self.norm_peak
			variance_channel = False
			# if psf.shape[-1] == 4:
			# 	variance_channel = True
			# 	variance_ch = psf[:, :, 2:]
			# 	var_total, std_total, std_ch1, std_ch2, var_ch1, var_ch2 = self.var_analysis(variance_ch)
			psfs_norm = np.sqrt(np.square(psfs_cos) + np.square(psfs_sin))
			psfs_z = np.arccos(np.divide(psfs_cos, psfs_norm + self.eps)) / np.pi
			psfs_logits = psfs_norm
			psfs_norm_clean = np.where(psfs_norm <= self.threshold_clean, 0, psfs_norm)
			label, features = nd_label(psfs_norm_clean)
			local_maximals = peak_local_max(psfs_norm, threshold_abs=self.threshold_abs, exclude_border=True,
											min_distance=self.threshold_distance, labels=label)
			candidates = []
			for count, local_maximal in enumerate(local_maximals):
				id_i, id_j = local_maximal
				if id_i >= self.i_skip_min and id_i <= self.i_skip_max and id_j >= self.j_skip_min and id_j <= self.j_skip_max:
					I_max = psfs_norm[id_i, id_j]
					I_sum = psfs_norm[id_i, id_j]
					I_sum += psfs_norm[id_i - 1, id_j]
					I_sum += psfs_norm[id_i + 1, id_j]
					I_sum += psfs_norm[id_i, id_j - 1]
					I_sum += psfs_norm[id_i, id_j + 1]
					prob_max = psfs_logits[id_i, id_j]
					prob_sum = psfs_logits[id_i - 1, id_j] + psfs_logits[id_i + 1, id_j] + psfs_logits[id_i, id_j - 1] + \
							   psfs_logits[id_i, id_j + 1] + psfs_logits[id_i, id_j]
					if I_max >= self.threshold_freq_max or I_sum >= self.threshold_freq_sum:
						dist_x = [psfs_norm[id_i - self.radius_lat, id_j], psfs_norm[id_i, id_j],
								  psfs_norm[id_i + self.radius_lat, id_j]]
						dist_y = [psfs_norm[id_i, id_j - self.radius_lat], psfs_norm[id_i, id_j],
								  psfs_norm[id_i, id_j + self.radius_lat]]
						x_correction = self.centroid([-1 * self.radius_lat, 0., self.radius_lat], dist_x,
													 self.x_px_size)
						y_correction = self.centroid([-1 * self.radius_lat, 0., self.radius_lat], dist_y,
													 self.y_px_size)
						x_px = (y_correction + id_j) / 4.
						y_px = (x_correction + id_i) / 4.
						z_weights = psfs_norm[id_i - self.radius_axi:id_i + self.radius_axi + 1,
									id_j - self.radius_axi:id_j + self.radius_axi + 1]
						z_dist = psfs_z[id_i - self.radius_axi:id_i + self.radius_axi + 1,
								 id_j - self.radius_axi:id_j + self.radius_axi + 1]

						if z_weights.sum() == 0:
							z_pi = 0
						else:
							z_pi = np.average(z_dist, weights=z_weights)
						z_nm = (z_pi * self.z_range) - (0.5000 * self.z_range)
						# if variance_channel:
						# 	crop = 1
						# 	var_cos_peak = np.sum(var_ch1[id_i - crop:id_i + crop + 1, id_j - crop:id_j + crop + 1])
						# 	var_sin_peak = np.sum(var_ch2[id_i - crop:id_i + crop + 1, id_j - crop:id_j + crop + 1])
						# 	var_peak = np.sum(var_total[id_i - crop:id_i + crop + 1, id_j - crop:id_j + crop + 1])
						# else:
						# 	var_cos_peak = 0
						# 	var_sin_peak = 0
						# 	var_peak = 0
						candidates.append({
							'X_pr_px': x_px,
							'Y_pr_px': y_px,
							'X_pr_nm': x_px * self.x_px_size,
							'Y_pr_nm': y_px * self.y_px_size,
							'Z_pr_nm': z_nm,
							'id_i': id_i,
							'id_j': id_j,
							# 'var_cos': var_cos_peak,
							# 'var_sin': var_sin_peak,
							# 'var_tot': var_peak,
							'prob_max': prob_max,
							'prob_sum': prob_sum,
							'Freq_max': I_max,
							'Freq_sum': I_sum})
			# uncertainty and probability of prediciton should be added here
			data_candids = pd.DataFrame(candidates)
			data_candids = data_candids.dropna()
			return data_candids
		except Exception as e:
			raise ValueError("Error in seed_candidate_3D: " + str(e))

	def filter_candidates(self, dataset):
		raise DeprecationWarning("This function is deprecated. Use seed_candidates_3D instead.")

	def localization_3D(self):
		global col_list
		try:
			# check gt file and consider matching is true/false
			gt = self.gt.copy()
			# assert seed_id and frame_id are integer

			loc_dataset = pd.DataFrame()  # Initialize an empty DataFrame to store results
			if isinstance(gt, list) and not gt:
				print("No gt file imported, then no matching will be applied.")
				for f in range(self.psfs.shape[0]):
					psf = self.psfs[f]
					Pr_frame = self.seed_candidate_3D(psf)
					if f == 0:
						col_list = list(Pr_frame.keys())
					Pr_frame['frame_id'] = f + 1
					Pr_frame['seed_id'] = list(np.arange(1, len(Pr_frame) + 1, dtype=np.int32))
					Pr_frame = Pr_frame[['frame_id', 'seed_id'] + col_list]  # adjust order of columns
					loc_dataset = pd.concat([loc_dataset, Pr_frame])  # Append the frame to all frames
			else:
				# print("gt file imported, matching will be applied automatically.")
				gt[['frame_id', 'seed_id']] = gt[['frame_id', 'seed_id']].astype('int')  # assert it is integer
				loc_dataset = pd.DataFrame()  # Initialize an empty DataFrame to store results
				for f in range(self.psfs.shape[0]):
					gtpr_frame = pd.DataFrame([])
					psf = self.psfs[f]
					frame_id = gt.frame_id.min() + f
					gt_frame = gt[gt.frame_id == frame_id]
					pr_Frame = self.seed_candidate_3D(psf)
					if gt_frame.empty and pr_Frame.empty:
						loc_dataset = pd.concat([loc_dataset, gtpr_frame])  # Append the frame to all frames
						continue
					labels = self.match_finder(pr_Frame, gt_frame)
					labels_TP = labels[labels.Condition == 'TP']
					labels_FP = labels[labels.Condition == 'FP']
					labels_FN = labels[labels.Condition == 'FN']
					seed_counter = 1
					for i in range(len(labels_TP)):
						seed_pr_id = labels_TP.pr_id.iloc[i]
						pr_seed = pr_Frame.iloc[seed_pr_id:seed_pr_id + 1, 0:]
						seed_tr_id = labels_TP.gt_id.iloc[i]
						gt_seed = gt_frame.iloc[seed_tr_id:seed_tr_id + 1, 0:]
						gt_seed['seed_id'] = seed_counter
						pr_seed['label'] = 'TP'
						gtpr_seed = gt_seed.join(pr_seed, how='cross')
						gtpr_frame = pd.concat([gtpr_frame, gtpr_seed])
						seed_counter += 1
					for j in range(len(labels_FN)):
						seed_tr_id = labels_FN.gt_id.iloc[j]
						gt_seed = gt_frame.iloc[seed_tr_id:seed_tr_id + 1, 0:]
						gt_seed['seed_id'] = seed_counter
						gt_seed['label'] = 'FN'
						gtpr_frame = pd.concat([gtpr_frame, gt_seed])
						seed_counter += 1
					for k in range(len(labels_FP)):
						seed_pr_id = labels_FP.pr_id.iloc[k]
						pr_seed = pr_Frame.iloc[seed_pr_id:seed_pr_id + 1, 0:]
						pr_seed['label'] = 'FP'
						pr_seed['frame_id'] = frame_id
						pr_seed['seed_id'] = 0
						gtpr_frame = pd.concat([gtpr_frame, pr_seed])
					gtpr_frame.reset_index(drop=True, inplace=True)
					loc_dataset = pd.concat([loc_dataset, gtpr_frame])  # Append the frame to all frames
			if self.save:
				path = os.path.join(os.getcwd(), 'log')
				if not os.path.exists(path):
					os.mkdir(path)
				unique_name = generate_unique_filename(loc_dataset.frame_id.max(), prefix="localization_result",
													   extension=".csv")
				saved_directory = os.path.join(path, unique_name)
				loc_dataset.to_csv(saved_directory, index=True)
				print()
				print('localization is done. File has been saved in log directory')
			return loc_dataset
		except Exception as e:
			# Handle exceptions and provide an error message
			raise ValueError("Error in localization_3D: " + str(e))


class localizer_machine_with_unc_model:
	def __init__(self, input_frame, psfs, model_unc, gt, param, crop_size=5):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.input_frame = input_frame
		self.sub_region_localizer = model_unc.to(self.device)
		self.sub_region_localizer.eval()
		self.crop_size = crop_size
		self.psfs_tensor = psfs.copy()
		self.psfs = np.moveaxis(psfs.cpu().numpy(), 1, -1)
		self.gt = gt
		self.gt['frame_id'].astype('int')
		self.gt['seed_id'].astype('int')
		self.param = param
		self.skip = self.param.post_processing.localization.skip
		self.norm_peak = self.param.Simulation.scale_factor
		self.z_range = self.param.Simulation.z_range
		self.threshold_freq_sum = self.param.post_processing.localization.threshold_freq_sum
		self.threshold_freq_max = self.param.post_processing.localization.threshold_freq_max
		self.threshold_clean = param.post_processing.localization.threshold_clean
		self.threshold_abs = param.post_processing.localization.threshold_abs
		self.threshold_distance = param.post_processing.localization.threshold_distance
		self.radius_lat = param.post_processing.localization.radius_lat
		self.radius_axi = param.post_processing.localization.radius_axi
		self.eps = param.post_processing.localization.epsilon
		if self.skip:
			self.i_skip_min = int(4 * param.post_processing.domain_pool[0][0])
			self.i_skip_max = int(4 * param.post_processing.domain_pool[0][1])
			self.j_skip_min = int(4 * param.post_processing.domain_pool[1][0])
			self.j_skip_max = int(4 * param.post_processing.domain_pool[1][1])
		else:
			self.i_skip_min = max(self.radius_lat, self.radius_axi)
			self.i_skip_max = 255 - max(self.radius_lat, self.radius_axi)
			self.j_skip_min = max(self.radius_lat, self.radius_axi)
			self.j_skip_max = 255 - max(self.radius_lat, self.radius_axi)
		self.x_px_size = param.Camera.px_size[0]
		self.y_px_size = param.Camera.px_size[1]
	@staticmethod
	def match_finder(pr_frame, gt_frame):
		if pr_frame.empty:
			gt_id = gt_frame.seed_id.to_list()
			Condition = ['FN'] * len(gt_id)
			pr_id = ['NA'] * len(gt_id)
			Res_data = pd.DataFrame({'Condition': Condition, 'gt_id': gt_id, 'pr_id': pr_id})
			return Res_data
		elif gt_frame.empty:
			pr_id = list(np.arange(len(pr_frame)))
			Condition = ['FP'] * len(pr_id)
			gt_id = ['NA'] * len(pr_id)
			Res_data = pd.DataFrame({'Condition': Condition, 'gt_id': gt_id, 'pr_id': pr_id})
			return Res_data
		else:
			pred_xyz = pr_frame[['X_pr_nm', 'Y_pr_nm', 'Z_pr_nm']].to_numpy()
			true_xyz = gt_frame[['X_tr_nm', 'Y_tr_nm', 'Z_tr_nm']].to_numpy()
			# Compute 2D and 3D distances
			cost2D = cdist(true_xyz[:, :2], pred_xyz[:, :2], 'euclidean')
			cost3D = cdist(true_xyz, pred_xyz, 'euclidean')
			# Find matching pairs based on 3D distances
			dist_matrix = np.sqrt(np.sqrt(cost3D))
			tr_id, pr_id = linear_sum_assignment(dist_matrix)

			# Initialize results list
			Results = []

			# Match TP, FN, and FP
			paired_tr = []
			paired_pr = []
			for i in range(len(tr_id)):
				if cost2D[tr_id[i], pr_id[i]] <= 250. and abs(true_xyz[tr_id[i], 2] - pred_xyz[pr_id[i], 2]) <= 500.:
					ids = {'Condition': 'TP', 'gt_id': tr_id[i], 'pr_id': pr_id[i]}
					paired_tr.append(tr_id[i])
					paired_pr.append(pr_id[i])
					Results.append(ids)

			all_tr_ids = list(np.arange(len(true_xyz)))
			T_Fals_id = list(set(all_tr_ids) - set(paired_tr))

			all_pr_ids = list(np.arange(len(pred_xyz)))
			P_Fals_id = list(set(all_pr_ids) - set(paired_pr))

			T_Fals_id.sort()
			for tt in T_Fals_id:
				ids = {'Condition': 'FN', 'gt_id': tt, 'pr_id': 'NA'}
				Results.append(ids)

			P_Fals_id.sort()
			for pp in P_Fals_id:
				ids = {'Condition': 'FP', 'gt_id': 'NA', 'pr_id': pp}
				Results.append(ids)

			# Convert results to DataFrame
			Res_data = pd.DataFrame(Results)
			return Res_data

	def seed_candidate_3D(self, psf, psf_tensor, input_frame):
		psfs_cos = psf[:, :, 0] / self.norm_peak
		psfs_sin = psf[:, :, 1] / self.norm_peak
		psfs_norm = np.sqrt(np.square(psfs_cos) + np.square(psfs_sin))
		psfs_z = np.arccos(np.divide(psfs_cos, psfs_norm + self.eps)) / np.pi
		psfs_logits = psfs_norm
		psfs_norm_clean = np.where(psfs_norm <= self.threshold_clean, 0, psfs_norm)
		label, features = nd_label(psfs_norm_clean)
		local_maximals = peak_local_max(psfs_norm, threshold_abs=self.threshold_abs, exclude_border=True,
										min_distance=self.threshold_distance, labels=label)

		candidates = []
		for count, local_maximal in enumerate(local_maximals):
			id_i, id_j = local_maximal
			if id_i >= self.i_skip_min and id_i <= self.i_skip_max and id_j >= self.j_skip_min and id_j <= self.j_skip_max:
				I_max = psfs_norm[id_i, id_j]
				I_sum = psfs_norm[id_i, id_j]
				I_sum += psfs_norm[id_i - 1, id_j]
				I_sum += psfs_norm[id_i + 1, id_j]
				I_sum += psfs_norm[id_i, id_j - 1]
				I_sum += psfs_norm[id_i, id_j + 1]
				prob_max = psfs_logits[id_i, id_j]
				prob_sum = psfs_logits[id_i - 1, id_j] + psfs_logits[id_i + 1, id_j] + psfs_logits[id_i, id_j - 1] + \
						   psfs_logits[id_i, id_j + 1] + psfs_logits[id_i, id_j]
				if I_max >= self.threshold_freq_max or I_sum >= self.threshold_freq_sum:
					psf_sub = psf_tensor[:, :, id_i - self.crop_size:id_i + self.crop_size + 1, id_j - self.crop_size:id_j + self.crop_size + 1]
					id_ix = int(id_i//4)
					id_jx = int(id_j//4)
					inp_sub = input_frame[:,:, id_ix - self.crop_size:id_ix + self.crop_size + 1, id_jx - self.crop_size:id_jx + self.crop_size + 1]
					psf_norm = torch.norm(psf_sub[:, 0:1, :, :], psf_sub[:, 1:2, :, :], dim=1)
					inp_z    = psfs_z[id_i - self.crop_size:id_i + self.crop_size + 1, id_j - self.crop_size:id_j + self.crop_size + 1]
					inp_z    = torch.tensor(inp_z).to(self.device).float()
					# add channel 0 and 1 to concat to inp_sub and psf_norm
					inp_z = inp_z.unsqueeze(0).unsqueeze(0)

					# concat all 1. psf_sub, 2. inp_sub, 3. psf_norm, 4. inp_z
					input_unc = torch.cat((psf_sub, inp_sub, psf_norm, inp_z), dim=1)
					xyz_corr = self.sub_region_localizer(input_unc)
					x_corr = xyz_corr[0, 0]
					y_corr = xyz_corr[0, 1]
					z_corr = xyz_corr[0, 2]
					x_std = xyz_corr[0, 3]
					y_std = xyz_corr[0, 4]
					z_std = xyz_corr[0, 5]

					x_px = (x_corr + id_j) / 4.
					y_px = (y_corr + id_i) / 4.
					z_nm = (z_corr * (self.z_range+0.2)) - (0.5000 * (self.z_range+0.2))
					candidates.append({
						'X_pr_px': x_px,
						'Y_pr_px': y_px,
						'X_pr_nm': x_px * self.x_px_size,
						'Y_pr_nm': y_px * self.y_px_size,
						'Z_pr_nm': z_nm,
						'id_i': id_i,
						'id_j': id_j,
						'x_std': x_std,
						'y_std': y_std,
						'z_std': z_std,
						'prob_max': prob_max,
						'prob_sum': prob_sum,
						'Freq_max': I_max,
						'Freq_sum': I_sum})
			data_candids = pd.DataFrame(candidates)
			data_candids = data_candids.dropna()
			return data_candids
	def localization_3D(self):
		gt = self.gt.copy()
		loc_dataset = pd.DataFrame()  # Initialize an empty DataFrame to store results
		for f in range(self.psfs.shape[0]):
			gtpr_frame = pd.DataFrame([])
			psf = self.psfs[f]
			psf_tensor = self.psfs_tensor[f:f+1]
			input_frame = self.input_frame[f:f+1]
			frame_id = gt.frame_id.min() + f
			gt_frame = gt[gt.frame_id == frame_id]
			pr_Frame = self.seed_candidate_3D(psf, psf_tensor, input_frame)
			if gt_frame.empty and pr_Frame.empty:
				loc_dataset = pd.concat([loc_dataset, gtpr_frame])  # Append the frame to all frames
				continue
			labels = self.match_finder(pr_Frame, gt_frame)
			labels_TP = labels[labels.Condition == 'TP']
			labels_FP = labels[labels.Condition == 'FP']
			labels_FN = labels[labels.Condition == 'FN']
			seed_counter = 1
			for i in range(len(labels_TP)):
				seed_pr_id = labels_TP.pr_id.iloc[i]
				pr_seed = pr_Frame.iloc[seed_pr_id:seed_pr_id + 1, 0:]
				seed_tr_id = labels_TP.gt_id.iloc[i]
				gt_seed = gt_frame.iloc[seed_tr_id:seed_tr_id + 1, 0:]
				gt_seed['seed_id'] = seed_counter
				pr_seed['label'] = 'TP'
				gtpr_seed = gt_seed.join(pr_seed, how='cross')
				gtpr_frame = pd.concat([gtpr_frame, gtpr_seed])
				seed_counter += 1
				for j in range(len(labels_FN)):
					seed_tr_id = labels_FN.gt_id.iloc[j]
					gt_seed = gt_frame.iloc[seed_tr_id:seed_tr_id + 1, 0:]
					gt_seed['seed_id'] = seed_counter
					gt_seed['label'] = 'FN'
					gtpr_frame = pd.concat([gtpr_frame, gt_seed])
					seed_counter += 1
					for k in range(len(labels_FP)):
						seed_pr_id = labels_FP.pr_id.iloc[k]
						pr_seed = pr_Frame.iloc[seed_pr_id:seed_pr_id + 1, 0:]
						pr_seed['label'] = 'FP'
						pr_seed['frame_id'] = frame_id
						pr_seed['seed_id'] = 0
						gtpr_frame = pd.concat([gtpr_frame, pr_seed])
						gtpr_frame.reset_index(drop=True, inplace=True)
						loc_dataset = pd.concat([loc_dataset, gtpr_frame])  # Append the frame to all frames
		return loc_dataset
