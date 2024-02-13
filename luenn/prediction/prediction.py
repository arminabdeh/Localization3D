import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def analyse(model, x_sim, param, progress_bar=True):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	x_sim = x_sim.to("cpu")
	batch_size = param.HyperParameter.batch_size
	dataset_test = torch.utils.data.TensorDataset(x_sim)
	dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
	model.eval()
	steps = len(dataloader_test)
	tqdm_enum = tqdm(total=steps, smoothing=0.)
	loc_file = None
	with torch.no_grad():
		for inputs in dataloader_test:
			outputs = model(inputs[0].to(device))
			inputs[0].cpu()
			temp_out = np.moveaxis(outputs.cpu().numpy(), 1, -1)
			loc_file = temp_out if loc_file is None else np.concatenate((loc_file, temp_out), axis=0)
			torch.cuda.empty_cache()
			if progress_bar:
				tqdm_enum.update(1)
	return loc_file
