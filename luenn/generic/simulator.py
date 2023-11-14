import decode
import numpy as np
import torch
from decode.neuralfitter.train.random_simulation import setup_random_simulation

from luenn.generic.label_generator import label_generator
from luenn.utils.utils import dec_luenn_gt_transform


class fly_simulator:
	def __init__(self,param,report=False):
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
		tar_em = tar_em[tar_em.phot>self.param.HyperParameter.emitter_label_photon_min]
		y_sim = label_generator(tar_em,frame_max,self.scale_factor,self.z_range,self.slide,self.box_size)
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
			print(f'Average seeds/frame is {total_seeds/total_frames}')
			print()
		return x_sim, y_sim, gt
	def ds_test(self):
		tar_em, sim_frames, bg_frames = self.sim_test.sample()
		frame_max = bg_frames.numpy().shape[0]
		tar_em = tar_em[tar_em.phot>self.param.HyperParameter.emitter_label_photon_min]
		y_sim = label_generator(tar_em,frame_max,self.scale_factor,self.z_range,self.slide,self.box_size)
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
			print(f'Average seeds/frame is {total_seeds/total_frames}')
			print()
		return x_sim, y_sim, gt

class validation_simulator:
	def __init__(self,param,report=False):
		self.report = report
		self.param = param
		self.param.HyperParameter.pseudo_ds_size = 1
		self.param.TestSet.test_size = 1
		self.n_frames = self.param.post_processing.simulation.validation_size
		sim_train, sim_test = setup_random_simulation(self.param)
		self.simulator = sim_test
		self.img_size = self.param.Simulation.img_size

	def sampling(self):
		n_min    = self.param.post_processing.simulation.n_min
		n_max    = self.param.post_processing.simulation.n_max
		Imean    = self.param.post_processing.simulation.Imean
		Isig     = self.param.post_processing.simulation.Isig
		x_min    =  self.param.post_processing.simulation.domain_pool[0][0]
		x_max    =  self.param.post_processing.simulation.domain_pool[0][1]
		y_min    =  self.param.post_processing.simulation.domain_pool[1][0]
		y_max    =  self.param.post_processing.simulation.domain_pool[1][1]
		z_min    =  self.param.post_processing.simulation.domain_pool[2][0]
		z_max    =  self.param.post_processing.simulation.domain_pool[2][1]
		px_size  = self.param.Camera.px_size
		ns = list(np.random.randint(n_min,n_max+1,size=self.n_frames))
		items = list(np.arange(0,self.n_frames,dtype=np.int32))
		frame_ix = torch.tensor([item for item, count in zip(items, ns) for _ in range(count)])
		prob = torch.tensor([1]*np.sum(ns)).float()
		ids = torch.tensor(list(np.arange(0,np.sum(ns),dtype=np.int32)))
		xyz = []
		phot = []
		for f in range(0,self.n_frames):
			N = ns[f]
			Is = np.random.normal(Imean,Isig, N)
			xs = np.random.uniform(x_min,x_max,N)
			ys = np.random.uniform(y_min,y_max,N)
			zs = np.random.uniform(z_min,z_max,N)
			for nn in range(N):
				xyz.append([xs[nn],ys[nn],zs[nn]])
				phot.append(max(Is[nn],1))
		xyz=torch.tensor(np.array(xyz)).float()
		phot=torch.tensor(np.array(phot)).float()
		tar_em = decode.EmitterSet(
			xyz=xyz,
			phot=phot,
			frame_ix=frame_ix,
			id = ids,
			prob = prob,
			xy_unit='px',
			px_size=(px_size[0], px_size[1]))
		return tar_em

	def sample(self):
		tar_em = self.sampling()
		x_sim = np.zeros((self.n_frames,1,int(self.img_size[0]),int(self.img_size[1])))
		for f in range(0,self.n_frames):
			em_frame  = tar_em[tar_em.frame_ix==f]
			xyzs      = em_frame.xyz
			intensity = em_frame.phot
			frame = self.simulator.psf.forward(xyzs,intensity)
			frame,bg = self.simulator.background.forward(frame)
			frame = self.simulator.noise.forward(frame)
			frame = frame.cpu()
			x_sim[f,0,:,:] +=np.array(frame[0,:,:]).T
		x_sim = torch.tensor(x_sim).float().cpu()
		gt = dec_luenn_gt_transform(tar_em)
		total_seeds = len(gt)
		total_frames = gt.frame_id.max()

		if self.report:
			print('Test data summary:')
			print(f'total seeds are {total_seeds}')
			print(f'total frames are {total_frames}')
			print(f'Average seeds/frame is {total_seeds/total_frames}')
			print()
		return x_sim,gt

