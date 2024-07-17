import decode.neuralfitter.train.live_engine
import decode.utils
import decode.utils
import h5py
import numpy as np
import pandas as pd
import torch
from luenn.utils.utils import dec_luenn_gt_transform


class decode_postprocessing:
	def __init__(self, Imean, Isig, N_frames, n_min, n_max, domain, model_dir, param_dir,save_dir=None):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.Imean = Imean
		self.Isig = Isig
		self.N_frames = N_frames
		self.n_min = n_min
		self.n_max = n_max
		self.param = decode.utils.param_io.load_params(param_dir)
		self.param.InOut.calibration_file = '/projects/academic/craigsno/arminabd/pycharm/luenn/config/calib/spline_calibration_3d_as_3dcal.mat'
		self.model = self._load_model(model_dir)
		self.sim_train, self.sim_test = decode.neuralfitter.train.live_engine.setup_random_simulation(self.param)
		self.domain_pool = domain
		self.save_dir = save_dir
	def hdf5_save(self, dataset):
		save_directory = self.save_dir+'input_frames.hdf5'
		fxy = h5py.File(save_directory, "w")
		fxy.create_dataset('inputs', data=dataset, compression="gzip", maxshape=(None,dataset.shape[1],dataset.shape[2]))
		fxy.close()
	def _load_model(self, model_dir):
		model = decode.neuralfitter.models.SigmaMUNet.parse(self.param)
		return decode.utils.model_io.LoadSaveModel(model, input_file=model_dir, output_file=None).load_init()

	def emitters_sampling(self):
		ns = list(np.array(abs(np.random.uniform(self.n_min, self.n_max, self.N_frames)), dtype=np.int64))
		x_min = self.domain_pool[0][0]
		x_max = self.domain_pool[0][1]
		y_min = self.domain_pool[1][0]
		y_max = self.domain_pool[1][1]
		z_min = self.domain_pool[2][0]
		z_max = self.domain_pool[2][1]

		xyz, phot, frame_ix, prob, ids = [], [], [], [], []
		i = 0
		p = 1.0

		for f in range(self.N_frames):
			N = max(int(ns[f]), 1)
			Is = np.random.normal(self.Imean, self.Isig, N)
			xs = np.random.uniform(x_min, x_max, N)
			ys = np.random.uniform(y_min, y_max, N)
			zs = np.random.uniform(z_min, z_max, N)
			for nn in range(N):
				frame_ix.append(f)
				xyz.append([xs[nn], ys[nn], zs[nn]])
				phot.append(max(Is[nn], 1))
				prob.append(p)
				ids.append(i)
				i += 1
		xyz = torch.tensor(np.array(xyz)).float()
		phot = torch.tensor(np.array(phot)).float()
		frame_ix = torch.tensor(np.array(frame_ix))
		ids = torch.tensor(np.array(ids))
		prob = torch.tensor(np.array(prob)).float()

		em = decode.EmitterSet(
			xyz=xyz,
			phot=phot,
			frame_ix=frame_ix,
			id=ids,
			prob=prob,
			xy_unit='px',
			px_size=(100., 100.)
		)
		return em

	def frame_simulation(self, GTs):
		img_size = self.param.Simulation.img_size
		N_frames = GTs.frame_ix.max() + 1
		x_sim_dec = np.zeros((N_frames, int(img_size[0]), int(img_size[1])))
		x_sim_lue = np.zeros((N_frames, 1, int(img_size[0]), int(img_size[1])))

		for f in range(N_frames):
			GTs_Frame = GTs[GTs.frame_ix == f]
			xyzs = GTs_Frame.xyz
			intensity = GTs_Frame.phot
			frame = self.sim_test.psf.forward(xyzs, intensity)
			frame_bg, bg = self.sim_test.background.forward(frame)
			frame_bg_n = self.sim_test.noise.forward(frame_bg)
			frame_bg_n = frame_bg_n.cpu()
			x_sim_dec[f, :, :] += np.array(frame_bg_n[0, :, :])
			x_sim_lue[f, 0, :, :] += np.array(frame_bg_n[0, :, :]).T

		x_sim_dec = torch.tensor(x_sim_dec).float()
		x_sim_lue = torch.tensor(x_sim_lue).float()
		return x_sim_dec, x_sim_lue

	def process_frames(self, frames):
		camera = decode.simulation.camera.Photon2Camera.parse(self.param)
		frame_proc = decode.neuralfitter.utils.processing.TransformSequence([decode.neuralfitter.scale_transform.AmplitudeRescale.parse(self.param)])
		size_procced = decode.neuralfitter.frame_processing.get_frame_extent(frames.unsqueeze(1).size(), frame_proc.forward)
		frame_extent = ((0.0, size_procced[-2]),(0.0, size_procced[-1]))
		post_proc = decode.neuralfitter.utils.processing.TransformSequence([decode.neuralfitter.scale_transform.InverseParamListRescale.parse(self.param),
			decode.neuralfitter.coord_transform.Offset2Coordinate(xextent=frame_extent[0], yextent=frame_extent[1], img_shape=size_procced[-2:]),
			decode.neuralfitter.post_processing.SpatialIntegration(raw_th=0.1, xy_unit='px', px_size=self.param.Camera.px_size)])
		infer = decode.neuralfitter.Infer(
			model=self.model,
			ch_in=self.param.HyperParameter.channels_in,
			frame_proc=frame_proc,
			post_proc=post_proc,
			device=self.device,
			batch_size='auto'
		)
		em_pred = infer.forward(frames)
		return em_pred

	def evaluate(self, em_pred, em_ref):
		matcher = decode.evaluation.match_emittersets.GreedyHungarianMatching(match_dims=3, dist_lat=250, dist_ax=500.)
		tp, fp, fn, tp_match = matcher.forward(em_pred, em_ref)
		tp_xyz_tr = em_ref.to_dict()['xyz'].tolist()
		tp_ref_id = em_ref.to_dict()['id'].tolist()
		frame_ix = em_ref.to_dict()['frame_ix'].tolist()
		tp_xyz_pr = tp.to_dict()['xyz'].tolist()
		fp_xyz_pr = fp.to_dict()['xyz'].tolist()
		fn_xyz_tr = fn.to_dict()['xyz'].tolist()
		tp_match_id = tp.to_dict()['id'].tolist()
		df_fp = pd.DataFrame({'frame_ix': -1,'ref_id': -1,
							  'X_pr_px': [x[0] for x in fp_xyz_pr],
							  'X_pr_nm': [x[0]*100.0 for x in fp_xyz_pr],
							  'Y_pr_px': [x[1] for x in fp_xyz_pr],
							  'Y_pr_nm': [x[1]*100.0 for x in fp_xyz_pr],
							  'Z_pr_nm': [x[2] for x in fp_xyz_pr],
							   'label': 'FP'})
		df_fn = pd.DataFrame({'frame_ix': -2,'ref_id': -2,
							  'X_tr_px': [x[0] for x in fn_xyz_tr],
							  'X_tr_nm': [x[0]*100.0 for x in fn_xyz_tr],
							  'Y_tr_px': [x[1] for x in fn_xyz_tr],
							  'Y_tr_nm': [x[1]*100.0 for x in fn_xyz_tr],
							  'Z_tr_nm': [x[2] for x in fn_xyz_tr],
							  'label': 'FN'})
		df_tp_pr = pd.DataFrame({'ref_id': tp_match_id,
								 'X_pr_px': [x[0] for x in tp_xyz_pr],
								 'X_pr_nm': [x[0]*100.0 for x in tp_xyz_pr],
								 'Y_pr_px': [x[1] for x in tp_xyz_pr],
								 'Y_pr_nm': [x[1]*100.0 for x in tp_xyz_pr],
								 'Z_pr_nm': [x[2] for x in tp_xyz_pr]})
		df_tp_tr = pd.DataFrame({'frame_ix': frame_ix,'ref_id': tp_ref_id,
								 'X_tr_px': [x[0] for x in tp_xyz_tr],
								 'X_tr_nm': [x[0]*100.0 for x in tp_xyz_tr],
								 'Y_tr_px': [x[1] for x in tp_xyz_tr],
								 'Y_tr_nm': [x[1]*100.0 for x in tp_xyz_tr],
								 'Z_tr_nm': [x[2] for x in tp_xyz_tr],
								 'label': 'TP'})
		result_df1 = pd.concat([df_fn, df_fp], axis=0)
		result_df2 = pd.merge(df_tp_tr, df_tp_pr, on='ref_id', how='inner')
		result_df = pd.concat([result_df2, result_df1], axis=0)
		return result_df

	def gt_transform(self, em_ref):
		gt_lue = dec_luenn_gt_transform(em_ref)
		# print(gt_lue)
		# gt_lue['X_tr_px'] += 0.5
		# gt_lue['Y_tr_px'] += 0.5
		# gt_lue['X_tr_nm'] += 50.0
		# gt_lue['Y_tr_nm'] += 50.0
		# print(gt_lue)
		return gt_lue
	def run_simulation(self):
		em_ref = self.emitters_sampling()
		gt_lue = self.gt_transform(em_ref)
		frames_dec,frames_lue = self.frame_simulation(em_ref)
		em_pred = self.process_frames(frames_dec)
		result_df = self.evaluate(em_pred, em_ref)
		if self.save_dir is not None:
			self.hdf5_save(frames_lue)
			gt_lue.to_csv(self.save_dir+'gt.csv',index=False)
		return em_ref, em_pred, frames_dec, result_df, gt_lue, frames_lue
