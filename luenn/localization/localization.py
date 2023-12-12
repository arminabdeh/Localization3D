import os
import numpy as np
import pandas as pd
from scipy.ndimage import label as nd_label
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from luenn.utils import generate_unique_filename
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.seterr(divide='ignore')
np.seterr(invalid='ignore')



class localizer_machine:
	def __init__(self, param, psfs, GT=None, save=None, photon_head=False):
		self.param = param
		self.photon_head = photon_head
		if self.photon_head:
			self.threshold_photons = self.param.post_processing.localization.threshold_photons
		else:
			self.threshold_photons = -1
		if isinstance(psfs, list):
			psfs = np.array(psfs)
		if torch.is_tensor(psfs):
			psfs = np.moveaxis(psfs.cpu().numpy(), 1, -1)
		psfs = psfs.astype(np.float64)
		self.psfs = psfs
		self.GT = GT
		if save is None:
			if self.param.post_processing.localization.save is None:
				self.save = False
			else:
				self.save = self.param.post_processing.localization.save
		else:
			self.save = save

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
		self.radius_phot = param.post_processing.localization.radius_phot
		self.eps = param.post_processing.localization.epsilon
		if self.skip:
			self.i_skip_min = int(4 * param.post_processing.domain_pool[0][0])
			self.i_skip_max = int(4 * param.post_processing.domain_pool[0][1])
			self.j_skip_min = int(4 * param.post_processing.domain_pool[1][0])
			self.j_skip_max = int(4 * param.post_processing.domain_pool[1][1])
		else:
			self.i_skip_min = max(self.radius_lat, self.radius_axi)
			self.i_skip_max = 255-max(self.radius_lat, self.radius_axi)
			self.j_skip_min = max(self.radius_lat, self.radius_axi)
			self.j_skip_max = 255-max(self.radius_lat, self.radius_axi)
		self.x_px_size = param.Camera.px_size[0]
		self.y_px_size = param.Camera.px_size[1]
	@staticmethod
	def centroid(x, w, px_size):
		w1 = w[0]
		w2 = w[1]
		w3 = w[2]
		x1 = x[0]
		x2 = x[1]
		x3 = x[2]
		# replace all ** with np.power
		t1 = np.power(w1, np.power(x3, 2) - np.power(x2, 2))
		t2 = np.power(w2, np.power(x1, 2) - np.power(x3, 2))
		t3 = np.power(w3, np.power(x2, 2) - np.power(x1, 2))
		t4 = np.power(w1, x3 - x2)
		t5 = np.power(w2, x1 - x3)
		t6 = np.power(w3, x2 - x1)
		mu = 0.5 * (np.log2(t1 * t2 * t3)) / np.log2(t4 * t5 * t6)
		sig = (max(0.5 * ((x1 - x3) * (x2 - x1) * (x3 - x2) / np.log2(t4 * t5 * t6)), 0)) ** .5
		sig_nm = (px_size / 4.) * abs(sig - 0.83065387)
		return mu, sig_nm

	@staticmethod
	def calculate_entropy(sigma_x, sigma_y, sigma_z):
		entropy2d = 0.5 * np.log2(2 * np.pi * np.e * sigma_x ** 2 * sigma_y ** 2)
		entropy3d = 0.5 * (np.log2((2 * np.pi * np.e) ** 3 * sigma_x * sigma_y * sigma_z))
		return entropy2d, entropy3d

	@staticmethod
	def match_finder(PR_Frame, GT_Frame):
		if PR_Frame.empty:
			GT_id = GT_Frame.seed_id.to_list()
			Condition = ['FN'] * len(GT_id)
			PR_id = ['NA'] * len(GT_id)
			Res_data = pd.DataFrame({'Condition': Condition, 'GT_id': GT_id, 'PR_id': PR_id})
			return Res_data
		elif GT_Frame.empty:
			PR_id = list(np.arange(len(PR_Frame)))
			Condition = ['FP'] * len(PR_id)
			GT_id = ['NA'] * len(PR_id)
			Res_data = pd.DataFrame({'Condition': Condition, 'GT_id': GT_id, 'PR_id': PR_id})
			return Res_data
		else:
			pred_xyz = PR_Frame[['X_pr_nm', 'Y_pr_nm', 'Z_pr_nm']].to_numpy()
			true_xyz = GT_Frame[['X_tr_nm', 'Y_tr_nm', 'Z_tr_nm']].to_numpy()
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
					ids = {'Condition': 'TP', 'GT_id': tr_id[i], 'PR_id': pr_id[i]}
					paired_tr.append(tr_id[i])
					paired_pr.append(pr_id[i])
					Results.append(ids)

			all_tr_ids = list(np.arange(len(true_xyz)))
			T_Fals_id = list(set(all_tr_ids) - set(paired_tr))

			all_pr_ids = list(np.arange(len(pred_xyz)))
			P_Fals_id = list(set(all_pr_ids) - set(paired_pr))

			T_Fals_id.sort()
			for tt in T_Fals_id:
				ids = {'Condition': 'FN', 'GT_id': tt, 'PR_id': 'NA'}
				Results.append(ids)

			P_Fals_id.sort()
			for pp in P_Fals_id:
				ids = {'Condition': 'FP', 'GT_id': 'NA', 'PR_id': pp}
				Results.append(ids)

			# Convert results to DataFrame
			Res_data = pd.DataFrame(Results)
			return pd.DataFrame(Results)

	def seed_candidate_3D(self, psf):
		try:
			# processing
			psfs_cos = psf[:, :, 0] / self.norm_peak
			psfs_sin = psf[:, :, 1] / self.norm_peak
			if self.photon_head:
				psfs_phot = psf[:, :, 2]
			else:
				intensity = 0
				psfs_phot = np.array([])
			psfs_norm = np.sqrt(np.square(psfs_cos) + np.square(psfs_sin))
			psfs_Z = np.arccos(np.divide(psfs_cos, psfs_norm + self.eps)) / np.pi
			psfs_norm_unit = psfs_norm
			psfs_norm_clean = np.where(psfs_norm <= self.threshold_clean, 0, psfs_norm)
			label, features = nd_label(psfs_norm_clean)
			local_maximals = peak_local_max(psfs_norm, threshold_abs=self.threshold_abs, exclude_border=True,
											min_distance=self.threshold_distance, labels=label)
			count_detected = len(local_maximals)
			candidates = []
			for local_maximal in local_maximals:
				Id_i, Id_j = local_maximal
				if Id_i >= self.i_skip_min and Id_i <= self.i_skip_max and Id_j >= self.j_skip_min and Id_j <= self.j_skip_max:
					I_max = psfs_norm_unit[Id_i, Id_j]
					I_sum = psfs_norm_unit[Id_i, Id_j]
					I_sum += psfs_norm_unit[Id_i - 1, Id_j]
					I_sum += psfs_norm_unit[Id_i + 1, Id_j]
					I_sum += psfs_norm_unit[Id_i, Id_j - 1]
					I_sum += psfs_norm_unit[Id_i, Id_j + 1]
					if I_max>=self.threshold_freq_max or I_sum>=self.threshold_freq_sum:

						Dist_X = [psfs_norm_unit[Id_i - self.radius_lat, Id_j], psfs_norm_unit[Id_i, Id_j],
								  psfs_norm_unit[Id_i + self.radius_lat, Id_j]]
						Dist_Y = [psfs_norm_unit[Id_i, Id_j - self.radius_lat], psfs_norm_unit[Id_i, Id_j],
								  psfs_norm_unit[Id_i, Id_j + self.radius_lat]]

						x_correction, sig_x = self.centroid([-1 * self.radius_lat, 0., self.radius_lat], Dist_X, self.x_px_size)
						y_correction, sig_y = self.centroid([-1 * self.radius_lat, 0., self.radius_lat], Dist_Y, self.y_px_size)
						X_px = (y_correction + Id_j) / 4.
						Y_px = (x_correction + Id_i) / 4.

						sig_z = self.z_range * np.std(psfs_Z[Id_i - self.radius_axi:Id_i + self.radius_axi + 1, 
							Id_j - self.radius_axi:Id_j + self.radius_axi + 1])

						entropy2d, entropy3d = self.calculate_entropy(sig_x, sig_y, sig_z)

						sig_xyz = np.sqrt(sig_x ** 2 + sig_y ** 2 + sig_z ** 2)

						z_weights = psfs_norm_unit[Id_i - self.radius_axi:Id_i + self.radius_axi + 1, Id_j - self.radius_axi:Id_j + self.radius_axi + 1]
						z_dist    = psfs_Z[Id_i - self.radius_axi:Id_i + self.radius_axi + 1, Id_j - self.radius_axi:Id_j + self.radius_axi + 1]
						# stop error of weights sum to zero can't e normalized
						if z_weights.sum() == 0:
							z_pi = 0
						else:
							z_pi = np.average(z_dist, weights=z_weights)
						Z_nm = (z_pi * self.z_range) - (0.5000 * self.z_range)
						if self.photon_head:
							phot_modif = list(psfs_phot[Id_i - self.radius_phot:Id_i + self.radius_phot + 1,
											  Id_j - self.radius_phot:Id_j + self.radius_phot + 1].flatten())
							phot_modif.sort()
							intensity = np.sum(phot_modif[-int(self.radius_phot ** 2):])
						candidates.append({
							'X_pr_px': X_px,
							'Y_pr_px': Y_px,
							'X_pr_nm': X_px * self.x_px_size,
							'Y_pr_nm': Y_px * self.y_px_size,
							'Z_pr_nm': Z_nm,
							'Id_i': Id_i,
							'Id_j': Id_j,
							'prob': I_max,
							'Freq_max': I_max,
							'Freq_sum': I_sum,
							'photons_pred': intensity,
							'sig_x': sig_x,
							'sig_y': sig_y,
							'sig_z': sig_z,
							'sig_xyz': sig_xyz,
							'entropy2d': entropy2d,
							'entropy3d': entropy3d})
					# uncertainty and probability of prediciton should be added here
			data_candids = pd.DataFrame(candidates)
			data_candids = data_candids.dropna()
			return data_candids
		except Exception as e:
			raise ValueError("Error in seed_candidate_3D: " + str(e))

	def filter_candidates(self, dataset):
		raise DeprecationWarning("This function is deprecated. Use seed_candidates_3D instead.")

	def localization_3D(self):
		try:
			# check gt file and consider matching is true/false
			GT = self.GT
			loc_dataset = pd.DataFrame()  # Initialize an empty DataFrame to store results
			if isinstance(GT, list) and not GT:
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
				GT[['frame_id', 'seed_id']] = GT[['frame_id', 'seed_id']].astype('int')  # assert it is integer
				loc_dataset = pd.DataFrame()  # Initialize an empty DataFrame to store results
				for f in range(self.psfs.shape[0]):
					GtPr_frame = pd.DataFrame([])
					psf = self.psfs[f]
					frame_id = GT.frame_id.min() + f
					GT_Frame = GT[GT.frame_id == frame_id]
					PR_Frame = self.seed_candidate_3D(psf)
					if GT_Frame.empty and PR_Frame.empty:
						loc_dataset = pd.concat([loc_dataset, GtPr_frame])  # Append the frame to all frames
						continue
					labels = self.match_finder(PR_Frame, GT_Frame)
					labels_TP = labels[labels.Condition == 'TP']
					labels_FP = labels[labels.Condition == 'FP']
					labels_FN = labels[labels.Condition == 'FN']
					seed_counter = 1
					for i in range(len(labels_TP)):
						seed_pr_id = labels_TP.PR_id.iloc[i]
						PR_seed = PR_Frame.iloc[seed_pr_id:seed_pr_id + 1, 0:]
						seed_tr_id = labels_TP.GT_id.iloc[i]
						GT_seed = GT_Frame.iloc[seed_tr_id:seed_tr_id + 1, 0:]
						GT_seed['seed_id'] = seed_counter
						PR_seed['label'] = 'TP'
						GtPr_seed = GT_seed.join(PR_seed, how='cross')
						GtPr_frame = pd.concat([GtPr_frame, GtPr_seed])
						seed_counter += 1
					for j in range(len(labels_FN)):
						seed_tr_id = labels_FN.GT_id.iloc[j]
						GT_seed = GT_Frame.iloc[seed_tr_id:seed_tr_id + 1, 0:]
						GT_seed['seed_id'] = seed_counter
						GT_seed['label'] = 'FN'
						GtPr_frame = pd.concat([GtPr_frame, GT_seed])
						seed_counter += 1
					for k in range(len(labels_FP)):
						seed_pr_id = labels_FP.PR_id.iloc[k]
						PR_seed = PR_Frame.iloc[seed_pr_id:seed_pr_id + 1, 0:]
						PR_seed['label'] = 'FP'
						PR_seed['frame_id'] = frame_id
						PR_seed['seed_id'] = 0
						GtPr_frame = pd.concat([GtPr_frame, PR_seed])
					GtPr_frame.reset_index(drop=True, inplace=True)
					loc_dataset = pd.concat(
						[loc_dataset, GtPr_frame])  # Append the frame to all frames
			if self.save:
				path = os.path.join(os.getcwd(), 'log')
				if not os.path.exists(path):
					os.mkdir(path)
				unique_name = generate_unique_filename(loc_dataset.frame_id.max(), prefix="localization_result", extension=".csv")
				saved_directory = os.path.join(path, unique_name)
				GtPr_frame.to_csv(saved_directory)
				print()
				print('localization is done. File has been saved in log directory')
			return loc_dataset
		except Exception as e:
			# Handle exceptions and provide an error message
			raise ValueError("Error in localization_3D: " + str(e))
