from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
def analyse(model,x_sim,param):
	batch_size = param.HyperParameter.batch_size
	dataset_test = torch.utils.data.TensorDataset(x_sim)
	dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,pin_memory=True)
	model.eval()
	steps = len(dataloader_test)
	tqdm_enum = tqdm(total=steps, smoothing=0.)
	loc_file = None
	with torch.no_grad():           
		for inputs in dataloader_test:
			outputs = model(inputs[0].cuda())
			inputs = inputs[0].cpu()
			temp_out = np.moveaxis(outputs.cpu().numpy(), 1, -1)
			loc_file = temp_out if loc_file is None else np.concatenate((loc_file, temp_out), axis=0)
			torch.cuda.empty_cache()
			tqdm_enum.update(1)
	return loc_file