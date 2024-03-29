import numpy as np


class reg_classification:
	def __init__(self, res_loc):
		self.res_loc = res_loc
		if res_loc.empty == True or 'TP' not in res_loc.label.unique():
			self.GT = 0
			self.TP = 0
			self.FP = 0
			self.FN = 0
			self.xe = 0
			self.ye = 0
			self.ze = 0
		else:
			self.TP = len(res_loc[res_loc.label == 'TP'])
			self.FP = len(res_loc[res_loc.label == 'FP'])
			self.FN = len(res_loc[res_loc.label == 'FN'])
			self.xe = res_loc[res_loc.label == 'TP'].X_tr_nm - res_loc[res_loc.label == 'TP'].X_pr_nm
			self.ye = res_loc[res_loc.label == 'TP'].Y_tr_nm - res_loc[res_loc.label == 'TP'].Y_pr_nm
			self.ze = res_loc[res_loc.label == 'TP'].Z_tr_nm - res_loc[res_loc.label == 'TP'].Z_pr_nm

	# classification
	def jaccardian_index(self):
		TPFNFP = self.TP + self.FN + self.FP
		if TPFNFP == 0:
			return 0.
		return self.TP * 100 / TPFNFP

	def recall(self):
		TPFN = self.TP + self.FN
		if TPFN == 0:
			return 0.
		return self.TP * 100 / TPFN

	def precision(self):
		TPFP = self.TP + self.FP
		if TPFP == 0:
			return 0.
		return self.TP * 100 / TPFP

	def f1_score(self):
		pre = self.precision()
		rec = self.recall()
		if pre + rec == 0:
			return 0.
		f1 = (2 * pre * rec) / (pre + rec)
		return f1

	# regression
	def rmse_3d(self):
		return np.sqrt(np.mean(((self.xe ** 2) + (self.ye ** 2) + (self.ze ** 2))))

	def rmse_2d(self):
		return np.sqrt(np.mean(((self.xe ** 2) + (self.ye ** 2))))

	def rmse_x(self):
		return np.sqrt(np.mean(self.xe ** 2))

	def rmse_y(self):
		return np.sqrt(np.mean(self.ye ** 2))

	def rmse_z(self):
		return np.sqrt(np.mean(self.ze ** 2))

	def std_x(self):
		return np.std(np.array(self.xe))

	def std_y(self):
		return np.std(np.array(self.ye))

	def std_z(self):
		return np.std(np.array(self.ze))

	def del_x(self):
		return np.mean(self.xe)

	def del_y(self):
		return np.mean(self.ye)

	def del_z(self):
		return np.mean(self.ze)

	def mae_x(self):
		return np.mean(np.abs(self.xe))

	def mae_y(self):
		return np.mean(np.abs(self.ye))

	def mae_z(self):
		return np.mean(np.abs(self.ze))

	def efficiency_lateral(self):
		t1 = (1 - (self.jaccardian_index() * 0.01)) ** 2
		t2 = ((self.rmse_2d() ** 2) * 2) * (.01 ** 2)
		return (1 - np.sqrt(t1 + t2)) * 100.

	def efficiency_axial(self):
		t1 = (1 - (self.jaccardian_index() * 0.01)) ** 2
		t2 = (self.rmse_z() ** 2) * (.005 ** 2)
		return (1 - np.sqrt(t1 + t2)) * 100.

	def efficiency_3d(self):
		t1 = self.efficiency_lateral()
		t2 = self.efficiency_axial()
		return (t1 + t2) / 2

	def subpixel_bias_err(self, steps=20.):
		df = self.res_loc.copy()
		df = df[df.label == 'TP']
		x_loc_float = df['X_tr_px'].astype('float')
		x_loc_int = df['X_tr_px'].astype('int')
		x_sub_loc = x_loc_float - x_loc_int - .5
		y_loc_float = df['Y_tr_px'].astype('float')
		y_loc_int = df['Y_tr_px'].astype('int')
		y_sub_loc = y_loc_float - y_loc_int - .5
		df['X_Subpixel_location'] = x_sub_loc
		df['Y_Subpixel_location'] = y_sub_loc
		mesh_xy_bin_size = int(100. / steps)
		mesh_xy = list(np.arange(-50, 51, mesh_xy_bin_size) / 100.)
		ratio = []
		for ii in range(0, len(mesh_xy) - 1):
			df_x = df[(df.X_Subpixel_location >= mesh_xy[ii]) & (df.X_Subpixel_location <= mesh_xy[ii + 1])]
			for jj in range(0, len(mesh_xy) - 1):
				df_xy = df_x[(df_x.Y_Subpixel_location >= mesh_xy[jj]) & (df_x.Y_Subpixel_location <= mesh_xy[jj + 1])]
				err_x = df_xy['X_tr_nm'] - df_xy['X_pr_nm']
				del_x = np.mean(err_x)
				err_y = df_xy['Y_tr_nm'] - df_xy['Y_pr_nm']
				del_y = np.mean(err_y)
				err_z = df_xy['Z_tr_nm'] - df_xy['Z_pr_nm']
				del_z = np.mean(err_z)
				tot_err_del = np.sqrt(del_x ** 2 + del_y ** 2 + del_z ** 2)
				tot_err_std = np.std(np.sqrt(err_x ** 2 + err_y ** 2 + err_z ** 2))
				ratio.append(tot_err_del / tot_err_std)
		ratio = np.array(ratio)
		ratio = ratio.reshape(int(ratio.shape[0] ** 0.5), int(ratio.shape[0] ** 0.5))
		return ratio

# class axial_recall(res_loc, step_z,param):
# 	self.res_loc = res_loc
# 	self.step_z = step_z
# 	self.param = param
# 	def stepwise_recall(self):
# 		processed_dataframe = self.res_loc.copy()
# 		res = []
# 		consolidated_Z = {'Z_min': [], 'Z_max': [], 'Z_ave': [], 'Recall': []}
# 		processed_dataframe_filter = processed_dataframe[
# 		(processed_dataframe['label'] == 'TP') | (processed_dataframe['label'] == 'FN')]
# 		axial_range = self.param.Simulation.axial_range
# 		l = int(axial_range / step_z)
# 		for zs in range(0, l):
# 			Z_tr_nm_min = (zs * step_z) - (axial_range / 2.)
# 			Z_tr_nm_max = Z_tr_nm_min + step_z
# 			processed_dataframe_Z = processed_dataframe_filter[
# 				(processed_dataframe_filter.Z_tr_nm >= Z_tr_nm_min) & (processed_dataframe_filter.Z_tr_nm < Z_tr_nm_max)]
# 			if processed_dataframe_Z.empty:
# 				recall_z = consolidated_Z.copy()
# 			else:
# 				re
# 				recall = self.reg
# 				classification(processed_dataframe_Z).recall()
# 				recall_z = consolidated_Z.copy()
# 				recall_z['Z_min'] = Z_tr_nm_min
# 				recall_z['Z_max'] = Z_tr_nm_max
# 				recall_z['Z_ave'] = (Z_tr_nm_min + Z_tr_nm_max) / 2.
# 				recall_z['Recall'] = recall
# 			res.append(recall_z)
# 		return pd.DataFrame(res)
# 	def consolidated_z_range(self):
# 		recalls_dense = self.stepwise_recall()
# 		axial_range = self.param.Simulation.axial_range
# 		Zmax_id = np.argmax(recalls_dense.Recall)
# 		Zmax    = recalls_dense.Z_ave.to_list()[Zmax_id]
# 		Rmax    = recalls_dense.Recall.to_list()[Zmax_id]
# 		res    = peak_widths(np.array(recalls_dense.Recall),[Zmax_id],rel_height=0.5)
# 		Z_step = recalls_dense.Z_ave.to_list()[1]- recalls_dense.Z_ave.to_list()[0]
# 		Z_min  = recalls_dense.Z_ave.to_list()[0]
# 		left_min_recall = recalls_dense.Recall[0:Zmax_id].min()
# 		right_min_recall = recalls_dense.Recall[Zmax_id:].min()
# 		left_sign = (Rmax/2.) - left_min_recall
# 		right_sign = (Rmax/2.) - right_min_recall
# 		if left_sign<=0.:
# 			zh_min = -(axial_range/2.)
# 		else:
# 			zh_min = ((res[2]*Z_step)-(axial_range/2.))[0]
# 		if right_sign<=0.:
# 			zh_max = axial_range/2.
# 		else:
# 			zh_max = ((res[3]*Z_step)-(axial_range/2.))[0]
# 		FWHM = zh_max-zh_min
# 		ConsZR = FWHM*Rmax
# 		return Zmax, Rmax, FWHM, ConsZR
#
