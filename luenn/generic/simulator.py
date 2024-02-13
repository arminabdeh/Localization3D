import decode
import numpy as np
import torch
from decode.neuralfitter.train.random_simulation import setup_random_simulation
from luenn.generic import label_generator
from luenn.utils.utils import dec_luenn_gt_transform
import pandas as pd


class fly_simulator:
	def __init__(self, param, report=False):
		self.report = report
		self.param = param
		sim_train, sim_test = setup_random_simulation(self.param)
		self.sim_train = sim_train
		self.sim_test = sim_test
		self.scale_factor = self.param.Simulation.scale_factor
		self.z_range = self.param.Simulation.z_range
		self.slide = self.param.Simulation.label_slide
		self.box_size = self.param.Simulation.dist_sq_size

	def ds_train(self):
		tar_em, sim_frames, bg_frames = self.sim_train.sample()
		frame_max = bg_frames.numpy().shape[0]
		tar_em = tar_em[tar_em.phot > self.param.HyperParameter.emitter_label_photon_min]
		y_sim = label_generator(tar_em, frame_max, self.scale_factor, self.z_range, self.slide,self.box_size,)
		sim_frames = torch.transpose(sim_frames, 2, 1)
		x_sim = torch.unsqueeze(sim_frames, 1)
		x_sim = x_sim.cpu()
		gt = dec_luenn_gt_transform(tar_em)
		total_seeds = len(gt)
		total_frames = gt.frame_id.max()
		if self.report:
			print('Train data summary:')
			print(f'total seeds are {total_seeds}')
			print(f'total frames are {total_frames}')
			print(f'Average seeds/frame is {total_seeds / total_frames}')
			print()
		return x_sim, y_sim, gt

	def ds_test(self):
		tar_em, sim_frames, bg_frames = self.sim_test.sample()
		frame_max = bg_frames.numpy().shape[0]
		tar_em = tar_em[tar_em.phot > self.param.HyperParameter.emitter_label_photon_min]
		y_sim = label_generator(tar_em, frame_max, self.scale_factor, self.z_range, self.slide,
								self.box_size)
		sim_frames = torch.transpose(sim_frames, 2, 1)
		x_sim = torch.unsqueeze(sim_frames, 1)
		x_sim = x_sim.cpu()

		gt = dec_luenn_gt_transform(tar_em)
		total_seeds = len(gt)
		total_frames = gt.frame_id.max()
		if self.report:
			print('Test data summary:')
			print(f'total seeds are {total_seeds}')
			print(f'total frames are {total_frames}')
			print(f'Average seeds/frame is {total_seeds / total_frames}')
			print()
		return x_sim, y_sim, gt


class modified_fly_simulator:
	def __init__(self, param, report=False):
		self.report = report
		self.param = param
		self.train_size = param.HyperParameter.pseudo_ds_size
		self.test_size = param.TestSet.test_size
		sim_train, sim_test = setup_random_simulation(self.param)
		self.simulator = sim_train
		self.scale_factor = self.param.Simulation.scale_factor
		self.z_range = self.param.Simulation.z_range
		self.slide = self.param.Simulation.label_slide
		self.box_size = self.param.Simulation.dist_sq_size

	def sampling(self, train_set=True):
		if train_set:
			n_frames = self.train_size
		else:
			n_frames = self.test_size
		n_mean = self.param.Simulation.emitter_av
		n_std = n_mean / 4
		ns = np.random.normal(n_mean, n_std, n_frames)
		ns = np.round(ns).astype(int)
		ns = list(np.clip(ns, 1))
		Imean = self.param.Simulation.intensity_mu_sig[0]
		Isig = self.param.Simulation.intensity_mu_sig[1]
		x_min = self.param.Simulation.emitter_extent[0][0]
		x_max = self.param.Simulation.emitter_extent[0][1]
		y_min = self.param.Simulation.emitter_extent[1][0]
		y_max = self.param.Simulation.emitter_extent[1][1]
		z_min = self.param.Simulation.emitter_extent[2][0]
		z_max = self.param.Simulation.emitter_extent[2][1]
		px_size = self.param.Camera.px_size
		items = list(np.arange(0, n_frames, dtype=np.int32))
		frame_ix = torch.tensor([item for item, count in zip(items, ns) for _ in range(count)])
		frame_ix = frame_ix.long()
		prob = torch.tensor([1] * np.sum(ns)).float()
		ids = torch.tensor(list(np.arange(0, np.sum(ns), dtype=np.int32)))
		ids = ids.long()
		xyz = []
		phot = []
		for f in range(0, n_frames):
			N = ns[f]
			Is = np.random.normal(Imean, Isig, N)
			xs = np.random.uniform(x_min, x_max, N)
			ys = np.random.uniform(y_min, y_max, N)
			zs = np.random.uniform(z_min, z_max, N)
			for nn in range(N):
				xyz.append([xs[nn], ys[nn], zs[nn]])
				phot.append(max(Is[nn], 1))
		xyz = torch.tensor(np.array(xyz)).float()
		phot = torch.tensor(np.array(phot)).float()
		tar_em = decode.EmitterSet(
			xyz=xyz,
			phot=phot,
			frame_ix=frame_ix,
			id=ids,
			xy_unit='px',
			prob=prob,
			px_size=(px_size[0], px_size[1]))
		return tar_em

	def generate_engine(self, tar_em, mode='train'):
		f_max = tar_em.frame_ix.max() + 1
		x_sim = np.zeros((f_max, 1, int(self.param.Simulation.img_size[0]), int(self.param.Simulation.img_size[1])))
		for f in range(0, f_max):
			em_frame = tar_em[tar_em.frame_ix == f]
			xyzs = em_frame.xyz
			intensity = em_frame.phot
			frame = self.simulator.psf.forward(xyzs, intensity)
			frame, bg = self.simulator.background.forward(frame)
			frame = self.simulator.noise.forward(frame)
			frame = frame.cpu()
			x_sim[f, 0, :, :] += np.array(frame[0, :, :]).T
		x_sim = torch.tensor(x_sim).float().cpu()
		gt = dec_luenn_gt_transform(tar_em)
		total_seeds = len(gt)
		if self.report:
			print(f'{mode} data summary:')
			print(f'total seeds are {total_seeds}')
			print(f'total frames are {f_max}')
			print(f'Average seeds/frame is {total_seeds / f_max}')
			# number of unique seeds
			gt_filter = gt.drop_duplicates(subset=['X_tr_px', 'Y_tr_px'])
			n_unique_seeds = len(gt_filter)
			average_photons = np.mean(gt.photons.values)
			print(f'Number of unique seeds is {n_unique_seeds}')
			print(f'Average photons per set {average_photons}')
			print('-' * 50)
		y_sim = label_generator(tar_em, f_max,
								self.scale_factor, self.z_range,
								self.slide, self.box_size)
		return x_sim, y_sim, gt

	def ds_train(self):
		tar_em = self.sampling(train_set=True)
		x, y, gt = self.generate_engine(tar_em, mode='train')
		return x, y, gt

	def ds_test(self):
		tar_em = self.sampling(train_set=False)
		x, y, gt = self.generate_engine(tar_em, mode='test')
		return x, y, gt


class validation_simulator:
	def __init__(self, param, report=False, with_label=False):
		self.with_label = with_label
		self.report = report
		self.param = param
		self.param.HyperParameter.pseudo_ds_size = 1
		self.param.TestSet.test_size = 1
		self.n_frames = self.param.post_processing.simulation.validation_size
		sim_train, sim_test = setup_random_simulation(self.param)
		self.simulator = sim_test
		self.img_size = self.param.Simulation.img_size
		if self.with_label:
			self.scale_factor = self.param.Simulation.scale_factor
			self.slide = self.param.Simulation.label_slide
			self.box_size = self.param.Simulation.dist_sq_size
			self.z_range = self.param.Simulation.z_range

	def sampling(self):
		n_min = self.param.post_processing.simulation.n_min
		n_max = self.param.post_processing.simulation.n_max
		Imean = self.param.Simulation.intensity_mu_sig[0]
		Isig = self.param.Simulation.intensity_mu_sig[1]
		x_min = self.param.post_processing.simulation.domain_pool[0][0]
		x_max = self.param.post_processing.simulation.domain_pool[0][1]
		y_min = self.param.post_processing.simulation.domain_pool[1][0]
		y_max = self.param.post_processing.simulation.domain_pool[1][1]
		z_min = self.param.post_processing.simulation.domain_pool[2][0]
		z_max = self.param.post_processing.simulation.domain_pool[2][1]
		px_size = self.param.Camera.px_size
		ns = list(np.random.randint(n_min, n_max + 1, size=self.n_frames))
		items = list(np.arange(0, self.n_frames, dtype=np.int32))
		frame_ix = torch.tensor([item for item, count in zip(items, ns) for _ in range(count)])
		frame_ix = frame_ix.long()
		prob = torch.tensor([1] * np.sum(ns)).float()
		ids = torch.tensor(list(np.arange(0, np.sum(ns), dtype=np.int32)))
		ids = ids.long()
		xyz = []
		phot = []
		for f in range(0, self.n_frames):
			N = ns[f]
			Is = np.random.normal(Imean, Isig, N)
			xs = np.random.uniform(x_min, x_max, N)
			ys = np.random.uniform(y_min, y_max, N)
			zs = np.random.uniform(z_min, z_max, N)
			for nn in range(N):
				xyz.append([xs[nn], ys[nn], zs[nn]])
				phot.append(max(Is[nn], 1))
		xyz = torch.tensor(np.array(xyz)).float()
		phot = torch.tensor(np.array(phot)).float()
		tar_em = decode.EmitterSet(
			xyz=xyz,
			phot=phot,
			frame_ix=frame_ix,
			id=ids,
			xy_unit='px',
			prob=prob,
			px_size=(px_size[0], px_size[1]))
		return tar_em

	def sample(self):
		tar_em = self.sampling()
		x_sim = np.zeros((self.n_frames, 1, int(self.img_size[0]), int(self.img_size[1])))
		for f in range(0, self.n_frames):
			em_frame = tar_em[tar_em.frame_ix == f]
			xyzs = em_frame.xyz
			intensity = em_frame.phot
			frame = self.simulator.psf.forward(xyzs, intensity)
			frame, bg = self.simulator.background.forward(frame)
			frame = self.simulator.noise.forward(frame)
			frame = frame.cpu()
			x_sim[f, 0, :, :] += np.array(frame[0, :, :]).T
		x_sim = torch.tensor(x_sim).float().cpu()
		gt = dec_luenn_gt_transform(tar_em)
		total_seeds = len(gt)
		total_frames = gt.frame_id.max()

		if self.report:
			print('Test data summary:')
			print(f'total seeds are {total_seeds}')
			print(f'total frames are {total_frames}')
			print(f'Average seeds/frame is {total_seeds / total_frames}')
			print()
		if self.with_label:
			y_sim = label_generator(tar_em, total_frames,
									self.scale_factor, self.z_range,
									self.slide, self.box_size)
		else:
			y_sim = None

		return x_sim, y_sim, gt


if __name__ == '__main__':
	from luenn.utils import param_reference, complex_real_map
	import matplotlib.pyplot as plt

	param = param_reference()
	x, y, gt = fly_simulator(param_reference(), report=True).ds_train()
	intens = gt.photons.values
	plt.hist(intens, bins=100)
	plt.show()

	x, y, gt = validation_simulator(param, report=True, with_label=True).sample()
	print('Test validation data simulator based on the param_reference:')
	print(f"validation set size is {x.shape[0]}")
	if y is not None:
		print(y.shape)
	print('Ground truth:')
	print(gt)
	print(y.shape)
	fig, ax = plt.subplots(2, 2, figsize=(15, 15))
	ax[0][0].imshow(x.cpu()[0, 0, :, :])
	ax[0][0].set_title('x_sim')
	if y is not None:
		ymap = complex_real_map(y.cpu())
		ax[0][1].imshow(ymap[0, :, :])
		ax[0][1].set_title('y_sim')
	seed_loc = gt[(gt.frame_id == 1) & (gt.seed_id == 1)]
	ym = int(seed_loc.X_tr_px.values[0])
	ym2 = int(seed_loc.X_tr_px.values[0] * 4 + .5)
	xm = int(seed_loc.Y_tr_px.values[0])
	xm2 = int(seed_loc.Y_tr_px.values[0] * 4 + .5)
	print(xm, ym, xm2, ym2)
	xbox = 6
	ybox = 3
	ax[1][0].imshow(x.cpu()[0, 0, xm - xbox:xm + xbox + 1, ym - xbox:ym + xbox + 1])
	ax[1][0].set_title('x_sim_zoom')
	if y is not None:
		ax[1][1].imshow(ymap[0, xm2 - ybox:xm2 + ybox + 1, ym2 - ybox:ym2 + ybox + 1])
		ax[1][1].set_title('y_sim_zoom')
		for i in range(ym2 - ybox, ym2 + ybox + 1):
			for j in range(xm2 - ybox, xm2 + ybox + 1):
				ax[1][1].text(i - (ym2 - ybox) - .5, j - (xm2 - ybox), str(np.round(ymap[0, j, i], 1)))
	plt.show()
