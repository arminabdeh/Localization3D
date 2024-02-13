from torch.utils.data import Dataset


class data_loader_stream(Dataset):
	def __init__(self, simulator, evaluate=False):
		if evaluate:
			self.data = simulator.ds_test()
		else:
			self.data = simulator.ds_train()

		self.num_frames = self.data[0].shape[0]

	def __len__(self):
		return self.num_frames

	def __getitem__(self, index):
		x_sim, y_sim, gt_sim = self.data
		x_sample = x_sim[index]
		y_sample = y_sim[index]
		return {'x': x_sample, 'y': y_sample}

	def get_all_gts(self):
		return self.data[2]
