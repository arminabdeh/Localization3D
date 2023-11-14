import numpy as np
import torch


def label_dist(xm, ym, box_size):
	mu = [2.+xm, 2.+ym]  
	sigma = [1.0, 1.0]  
	x, y = np.meshgrid(np.arange(box_size), np.arange(box_size))
	gaussian = np.exp(-((x - mu[0])**2 / (2 * sigma[0]**2) + (y - mu[1])**2 / (2 * sigma[1]**2)))
	gaussian_norm = gaussian/np.sum(gaussian)
	gaussian_unit = gaussian/np.max(gaussian)
	return gaussian_unit, gaussian_norm

def label_generator(tar_em, frame_max, scale_factor, z_range, slide, box_size):
	frame_ids =  tar_em.frame_ix.numpy() #start from 0 to F: total F+1
	frame_ids = list(np.unique(frame_ids))
	Y_train = np.zeros((frame_max, 3, 256, 256), dtype=np.float64)
	for f in frame_ids:
		GT_Frame = tar_em[tar_em.frame_ix == f]
		xyz = GT_Frame.xyz_px.tolist()
		phot = GT_Frame.phot.tolist()
		for n in range(len(xyz)):
			xf = xyz[n][1] * 4
			yf = xyz[n][0] * 4
			xi = int(xf + 0.5)
			yj = int(yf + 0.5)
			if slide:
				xm = xf-xi
				ym = yf-yj
				dist_unit,dist_norm = label_dist(xm,ym,box_size)

			else:
				dist_unit,dist_norm = label_dist(0., 0., box_size)
			if 2 < xi < 253 and 2 < yj < 253:
				z_true = xyz[n][2]
				zr = np.pi * ((z_true + (0.5 * z_range)) / z_range)
				
				channel_1 = np.array(scale_factor*dist_unit * np.cos(zr)) #cos
				channel_2 = np.array(scale_factor*dist_unit * np.sin(zr)) #sin
				channel_3 = dist_norm*phot[n] #intensity
				
				left_side  = box_size // 2
				right_side = left_side + 1
				
				Y_train[f, 0, xi - left_side:xi + right_side, yj - left_side:yj + right_side] += channel_1
				Y_train[f, 1, xi - left_side:xi + right_side, yj - left_side:yj + right_side] += channel_2
				Y_train[f, 2, xi - left_side:xi + right_side, yj - left_side:yj + right_side] += channel_3
	Y_train = torch.tensor(Y_train).float()
	return Y_train


