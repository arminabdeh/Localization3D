import numpy as np
import torch


def label_generator(tar_em, frame_max, scale_factor, z_range, slide, box_size):
	frame_ids = np.unique(tar_em.frame_ix.numpy())
	y_train = np.zeros((frame_max, 2, 256, 256), dtype=np.float64)
	for f in frame_ids:
		gt_frame = tar_em[tar_em.frame_ix == f]
		xyz = gt_frame.xyz_px.tolist()
		for n in range(len(xyz)):
			xf = xyz[n][1] * 4
			yf = xyz[n][0] * 4
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

			if slide:
				dist_unit = label_dist(xm, ym, box_size)
			else:
				dist_unit = label_dist(0., 0., box_size)

			if 2 <= xi < 254 - box_size and 2 <= yi < 254 - box_size:
				z_true = xyz[n][2]
				zr = np.pi * ((z_true + (0.5 * z_range)) / z_range)
				channel_1 = scale_factor * dist_unit * np.cos(zr)
				channel_2 = scale_factor * dist_unit * np.sin(zr)

				left_side = box_size // 2
				right_side = left_side + 1

				y_train[f, 0, xi - left_side:xi + right_side, yi - left_side:yi + right_side] = channel_1
				y_train[f, 1, xi - left_side:xi + right_side, yi - left_side:yi + right_side] = channel_2

	y_train = torch.tensor(y_train).float()
	return y_train


def label_dist(xm, ym, box_size):
	mu = [2 + ym, 2 + xm]
	sigma = [1.0, 1.0]
	xs = np.arange(0, 5)
	ys = np.arange(0, 5)
	x, y = np.meshgrid(xs, ys)
	gaussian = np.exp(-((x - mu[0]) ** 2 / (2 * sigma[0] ** 2) + (y - mu[1]) ** 2 / (2 * sigma[1] ** 2)))
	return gaussian
