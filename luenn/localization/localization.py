import os

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datetime import datetime

def generate_unique_filename(f,prefix="localization_result", extension=".csv"):
    timestamp = datetime.now().strftime("%Y.%m.%d")
    unique_filename = f"{prefix}_{timestamp}_{str(f)}xframe{extension}"
    return unique_filename


class localizer_machine:
	def __init__(self, param,psfs,GT=[]):
		self.param = param
		self.psfs = psfs
		self.GT = GT
		self.save  = self.param.post_processing.localization.save
		self.skip = self.param.post_processing.localization.skip
		self.norm_peak = self.param.Simulation.scale_factor
		self.z_range = self.param.Simulation.z_range	

	@staticmethod   
	def centroid(x,w,px_size):
		w1 = w[0]
		w2 = w[1]
		w3 = w[2]
		x1 = x[0]
		x2 = x[1]
		x3 = x[2]
		t1 = w1**((x3**2)-(x2**2))
		t2 = w2**((x1**2)-(x3**2))
		t3 = w3**((x2**2)-(x1**2))
		t4 = w1**(x3-x2)
		t5 = w2**(x1-x3)
		t6 = w3**(x2-x1)
		mu = 0.5*(np.log2(t1*t2*t3))/np.log2(t4*t5*t6)
		sig = (max(0.5*((x1-x3)*(x2-x1)*(x3-x2)/np.log2(t4*t5*t6)),0))**.5
		sig_nm = (px_size/4.)*abs(sig-0.83065387)
		return mu, sig_nm
	@staticmethod
	def calculate_entropy(sigma_x, sigma_y, sigma_z):
		entropy2d = 0.5 * np.log2(2 * np.pi * np.e * sigma_x**2 * sigma_y**2)
		entropy3d = 0.5 * (np.log2((2 * np.pi * np.e ) ** 3 * sigma_x * sigma_y * sigma_z))
		return entropy2d,entropy3d
	@staticmethod   
	def match_finder(PR_Frame,GT_Frame):
		if PR_Frame.empty:
			GT_id     = GT_Frame.seed_id.to_list()
			Condition = ['FN']*len(GT_id)
			PR_id     = ['NA']*len(GT_id)
			Res_data = pd.DataFrame({'Condition': Condition, 'GT_id': GT_id, 'PR_id': PR_id})
			return Res_data
		elif GT_Frame.empty:
			PR_id     = list(np.arange(len(PR_Frame)))
			Condition = ['FP']*len(PR_id)
			GT_id     = ['NA']*len(PR_id)
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
	def seed_candidate_3D(self,psf):
		try:
			# initialize
			if self.skip:
				i_skip_min =  int(4*self.param.post_processing.domain_pool[0][0])
				i_skip_max =  int(4*self.param.post_processing.domain_pool[0][1])
				j_skip_min =  int(4*self.param.post_processing.domain_pool[1][0])
				j_skip_max =  int(4*self.param.post_processing.domain_pool[1][1])
			else:
				i_skip_min =  2
				i_skip_max =  253
				j_skip_min =  2
				j_skip_max =  253
			x_px_size = self.param.Camera.px_size[0]
			y_px_size = self.param.Camera.px_size[1]
			threshold_clean       = self.param.post_processing.localization.threshold_clean
			threshold_abs         = self.param.post_processing.localization.threshold_abs
			threshold_distance    = self.param.post_processing.localization.threshold_distance
			radius_lat            = self.param.post_processing.localization.radius_lat
			radius_axi            = self.param.post_processing.localization.radius_axi
			radius_phot           = self.param.post_processing.localization.radius_phot
			eps                   = self.param.post_processing.localization.epsilon
			#processing
			psfs_cos = psf[:,:,0]
			psfs_sin = psf[:,:,1]
			psfs_phot = psf[:,:,2]
			psfs_norm  = np.sqrt(np.square(psfs_cos)+np.square(psfs_sin))
			psfs_Z    = np.arccos(np.divide(psfs_cos,psfs_norm+eps))/np.pi
			psfs_norm_unit = psfs_norm/self.norm_peak
			psfs_norm_clean = np.where(psfs_norm<=threshold_clean,0,psfs_norm)
			label, features = scipy.ndimage.label(psfs_norm_clean)
			local_maximals = peak_local_max(psfs_norm,threshold_abs=threshold_abs,exclude_border=True,min_distance=threshold_distance,labels=label)
			count_detected = len(local_maximals)
			candidates = []
			for local_maximal in local_maximals:
				Id_i, Id_j = local_maximal
				if Id_i>=i_skip_min and Id_i<=i_skip_max and Id_j>=j_skip_min and Id_j<=j_skip_max:
					I_max = psfs_norm_unit[Id_i,Id_j]
					I_sum =psfs_norm_unit[Id_i,Id_j]
					I_sum+= psfs_norm_unit[Id_i-1,Id_j]
					I_sum+=psfs_norm_unit[Id_i+1,Id_j]
					I_sum+=psfs_norm_unit[Id_i,Id_j-1]
					I_sum+=psfs_norm_unit[Id_i,Id_j+1]

					Dist_X = [psfs_norm_unit[Id_i-radius_lat,Id_j],psfs_norm_unit[Id_i,Id_j],psfs_norm_unit[Id_i+radius_lat,Id_j]]
					Dist_Y = [psfs_norm_unit[Id_i,Id_j-radius_lat],psfs_norm_unit[Id_i,Id_j],psfs_norm_unit[Id_i,Id_j+radius_lat]]
					x_correction,sig_x = self.centroid([-1*radius_lat,0.,radius_lat],Dist_X,x_px_size)
					y_correction,sig_y = self.centroid([-1*radius_lat,0.,radius_lat],Dist_Y,y_px_size)
					sig_z   = self.z_range*np.std(psfs_Z[Id_i-radius_axi:Id_i+radius_axi+1,Id_j-radius_axi:Id_j+radius_axi+1])
					entropy2d,entropy3d = self.calculate_entropy(sig_x,sig_y,sig_z)
					sig_xyz = np.sqrt(sig_x**2+sig_y**2+sig_z**2)
					X_px = (y_correction+Id_j)/4.
					Y_px = (x_correction+Id_i)/4.
					z_pi = np.average(psfs_Z[Id_i-radius_axi:Id_i+radius_axi+1,Id_j-radius_axi:Id_j+radius_axi+1],
						weights=psfs_norm_unit[Id_i-radius_axi:Id_i+radius_axi+1,Id_j-radius_axi:Id_j+radius_axi+1])
					Z_nm = (z_pi*self.z_range)-(0.5000*self.z_range)
					phot_modif = list(psfs_phot[Id_i-radius_phot:Id_i+radius_phot+1,Id_j-radius_phot:Id_j+radius_phot+1].flatten())
					phot_modif.sort()
					intensity = np.sum(phot_modif[-int(radius_phot**2):])

					candidates.append({
						'X_pr_px': X_px,
						'Y_pr_px': Y_px,
						'X_pr_nm': X_px*x_px_size,
						'Y_pr_nm': Y_px*y_px_size,
						'Z_pr_nm': Z_nm,
						'Id_i': Id_i,
						'Id_j': Id_j,
						'prob': I_max,
						'Freq_max': I_max,
						'Freq_sum': I_sum,
						'photons_pred': intensity,
						'sig_x':sig_x,
						'sig_y':sig_y,
						'sig_z':sig_z,
						'sig_xyz':sig_xyz,
						'entropy2d':entropy2d,
						'entropy3d':entropy3d})
					#uncertainty and probability of prediciton should be added here
				data_candids = pd.DataFrame(candidates)
			return data_candids
		except Exception as e:
			raise ValueError("Error in seed_candidate_3D: " + str(e))
	def filter_candidates(self,dataset):
		try:
			threshold_freq_sum    = self.param.post_processing.localization.threshold_freq_sum
			threshold_freq_max    = self.param.post_processing.localization.threshold_freq_max
			threshold_photons     = self.param.post_processing.localization.threshold_photons
			if len(dataset)>0:
				dataset        = dataset[dataset['photons_pred']>threshold_photons]
				dataset = dataset[(dataset['Freq_sum']>threshold_freq_sum)|(dataset['prob']>threshold_freq_max)]
			return dataset
		except Exception as e:
			# Handle exceptions and provide an error message
			raise ValueError("Error in filter_candidates: " + str(e))
	def localization_3D(self):
		try:
			# check psf format
			psfs = self.psfs
			if isinstance(psfs, list):
				psfs = np.array(psfs)
			if torch.is_tensor(psfs):
				psfs = np.moveaxis(psfs.cpu().numpy(), 1, -1)
			# check gt file and consider matching is true/false
			GT = self.GT
			loc_dataset = pd.DataFrame() # Initialize an empty DataFrame to store results
			if isinstance(GT, list) and not GT:
				print("No gt file imported, then no matching will be applied.")
				for f in range(psfs.shape[0]):
					psf=psfs[f]
					Pr_frame = self.seed_candidate_3D(psf)
					Pr_frame = self.filter_candidates(Pr_frame)
					if f==0:
						col_list = list(Pr_frame.keys())
					Pr_frame['frame_id'] = f+1
					Pr_frame['seed_id'] = list(np.arange(1,len(Pr_frame)+1,dtype=np.int32))
					Pr_frame = Pr_frame[['frame_id','seed_id']+col_list] #adjust order of columns
					loc_dataset = pd.concat([loc_dataset, Pr_frame])  # Append the frame to all frames
			else:
				print("gt file imported, matching will be applied automatically.")
				GT[['frame_id','seed_id']] = GT[['frame_id','seed_id']].astype('int') #assert it is integer
				loc_dataset = pd.DataFrame()  # Initialize an empty DataFrame to store results
				for f in range(psfs.shape[0]):
					GtPr_frame = pd.DataFrame([])
					psf=psfs[f]
					frame_id = GT.frame_id.min()+f
					GT_Frame = GT[GT.frame_id==frame_id]
					PR_Frame = self.seed_candidate_3D(psf)
					PR_Frame = self.filter_candidates(PR_Frame)
					if GT_Frame.empty and PR_Frame.empty:
						loc_dataset = pd.concat([loc_dataset, GtPr_frame])  # Append the frame to all frames
						continue
					labels = self.match_finder(PR_Frame,GT_Frame)
					labels_TP = labels[labels.Condition=='TP']
					labels_FP = labels[labels.Condition=='FP']
					labels_FN = labels[labels.Condition=='FN']
					seed_counter = 1
					for i in range(len(labels_TP)):
						seed_pr_id = labels_TP.PR_id.iloc[i]
						PR_seed = PR_Frame.iloc[seed_pr_id:seed_pr_id+1, 0:]
						seed_tr_id = labels_TP.GT_id.iloc[i]
						GT_seed = GT_Frame.iloc[seed_tr_id:seed_tr_id+1, 0:]
						GT_seed['seed_id'] = seed_counter
						PR_seed['label'] = 'TP'
						GtPr_seed = GT_seed.join(PR_seed, how='cross')
						GtPr_frame = pd.concat([GtPr_frame, GtPr_seed])
						seed_counter += 1
					for j in range(len(labels_FN)):
						seed_tr_id = labels_FN.GT_id.iloc[j]
						GT_seed = GT_Frame.iloc[seed_tr_id:seed_tr_id+1, 0:]
						GT_seed['seed_id'] = seed_counter
						GT_seed['label'] = 'FN'
						GtPr_frame = pd.concat([GtPr_frame, GT_seed])
						seed_counter += 1
					for k in range(len(labels_FP)):
						seed_pr_id = labels_FP.PR_id.iloc[k]
						PR_seed = PR_Frame.iloc[seed_pr_id:seed_pr_id+1, 0:]
						PR_seed['label'] = 'FP'
						PR_seed['frame_id'] = frame_id
						PR_seed['seed_id'] = 0
						GtPr_frame = pd.concat([GtPr_frame, PR_seed])    
					GtPr_frame.reset_index(drop=True, inplace=True)
					loc_dataset = pd.concat([loc_dataset, GtPr_frame])  # Append the frame to all frames                   
			if self.save:
				path = os.path.join(os.getcwd(), 'log')
				if not os.path.exists(path):
					os.mkdir(path)
				
				saved_directory = os.path.join(path, generate_unique_filename(loc_dataset.frame_id.max()))
				GtPr_frame.to_csv(saved_directory)
				print('localization is done. File has been saved in log directory')
			return loc_dataset
		except Exception as e:
			# Handle exceptions and provide an error message
			raise ValueError("Error in localization_3D: " + str(e))