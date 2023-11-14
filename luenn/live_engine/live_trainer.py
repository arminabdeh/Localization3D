import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
from luenn.utils import load_model
from torch.utils.data import DataLoader,Dataset
from luenn.generic import fly_simulator
from luenn.model.model import UNet
from torch.utils.tensorboard import SummaryWriter
from luenn.localization import localizer_machine
from luenn.evaluate import reg_classification
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from luenn.evaluate import reg_classification
from luenn.generic import fly_simulator
from luenn.localization import localizer_machine
from luenn.model.model import UNet
from luenn.utils import load_model


class training_stream(Dataset):
	def __init__(self, param,simulator):
		self.data = simulator.ds_train()
		self.num_frames = self.data[0].numpy().shape[0] 
	def __len__(self):
		return self.num_frames

	def __getitem__(self, index):
		x_sim, y_sim, gt_sim = self.data
		return x_sim[index], y_sim[index]



class live_trainer:
	def __init__(self, param):
		self.param = param
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = UNet()
		self.model.to(self.device)
		self.batch_size = param.HyperParameter.batch_size
		self.num_epochs = param.HyperParameter.epochs
		self.lr = param.HyperParameter.lr
		self.gamma = param.HyperParameter.learning_rate_scheduler_param.gamma
		self.step_size = param.HyperParameter.learning_rate_scheduler_param.step_size
		self.restart_epochs = param.HyperParameter.restart_period
		self.writer = SummaryWriter()
		self.model_save_path = param.InOut.model_out
		self.checkpoint_save_path = param.InOut.model_check
		self.model_load_path = param.InOut.model_in

		self.train_size = param.HyperParameter.pseudo_ds_size
		self.train_losses = []
		self.validation_losses = []
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss()
		self.metric_min = float('inf')
		self.epoch_update = 0
		self.metric_cur = 0
		self.scheduler1 = lr_scheduler.StepLR(self.optimizer, self.step_size,gamma=self.gamma)
		self.scheduler2 = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

	def train_one_epoch(self,dataloader_train):
		self.model.train()
		steps = len(dataloader_train)
		tqdm_enum = tqdm(total=steps, smoothing=0.)
		train_loss = 0
		for data in dataloader_train:
			inputs, labels = data
			inputs = inputs.cuda()
			labels = labels.cuda()
			self.optimizer.zero_grad()
			outputs = self.model(inputs)
			loss = self.criterion(outputs, labels)
			train_loss+=loss
			loss.backward()
			self.optimizer.step()
			tqdm_enum.update(1)
			inputs = inputs.cpu()
			labels = labels.cpu()
			torch.cuda.empty_cache()
		return train_loss/len(dataloader_train)

	def save_checkpoint(self,epoch_cur):
		torch.save({'epoch':epoch_cur,'model_state_dict':self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()},
			self.checkpoint_save_path)
		if self.metric_cur<self.metric_min:
			print()
			print(f'metric improved from {self.metric_min} to {self.metric_cur}')
			torch.save({'epoch':epoch_cur,'model_state_dict':self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()},
				self.model_save_path)
			self.metric_min = self.metric_cur
			self.epoch_update = epoch_cur
		else:
			print()
			print(f'current metric is {self.metric_cur}')
			print(f'metric did not improve from {self.metric_min} updated at epoch {self.epoch_update+1}')

	def validate(self,dataloader_test,gt_val,full_process=False):
		self.model.eval()
		steps = len(dataloader_test)
		tqdm_enum = tqdm(total=steps, smoothing=0.)
		val_loss = 0
		loc_file = None 
		pr_gt = pd.DataFrame([])
		with torch.no_grad():			
			for data in dataloader_test:
				inputs, labels = data
				outputs = self.model(inputs.cuda())
				loss = self.criterion(outputs, labels.cuda())
				val_loss += loss.item()
				tqdm_enum.update(1)
				inputs = inputs.cpu()
				labels = labels.cpu()
				torch.cuda.empty_cache()
				if full_process:
					temp_out = np.moveaxis(outputs.cpu().numpy(), 1, -1)
					loc_file = temp_out if loc_file is None else np.concatenate((loc_file, temp_out), axis=0)

		val_loss /=len(dataloader_test)
		if full_process:
			pr_gt = localizer_machine(self.param,loc_file,gt_val).localization_3D()
			print(pr_gt)
		return val_loss,pr_gt

	def train(self):
		if self.model_load_path:
			self.model = load_model(self.model_load_path)
			print('resume training')
		simulator = fly_simulator(self.param,report=True)
		x_test, y_test, gt_test = simulator.ds_test()
		dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
		dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False,pin_memory=True)

		#training data
		streaming_dataset = training_stream(self.param, simulator)
		dataloader_train = DataLoader(streaming_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
		for epoch in range(self.num_epochs):
			if epoch%self.restart_epochs==0:
				print('restart simulation setup')
				streaming_dataset = training_stream(self.param, simulator)
				dataloader_train = DataLoader(streaming_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
			loss_train = self.train_one_epoch(dataloader_train)
			if (epoch+1)%1==0:
				loss_val,pr_gt   = self.validate(dataloader_test,gt_test,full_process=True)
				ji     = reg_classification(pr_gt).jaccardian_index()
				rec    = reg_classification(pr_gt).recall()
				pre    = reg_classification(pr_gt).precision()
				f1     = reg_classification(pr_gt).f1_score()
				rmse3d = reg_classification(pr_gt).rmse_3d()
				rmse2d = reg_classification(pr_gt).rmse_2d()
				rmsex  = reg_classification(pr_gt).rmse_x()
				rmsey  = reg_classification(pr_gt).rmse_y()
				rmsez  = reg_classification(pr_gt).rmse_z()
				delx   = reg_classification(pr_gt).del_x()
				dely   = reg_classification(pr_gt).del_y()
				delz   = reg_classification(pr_gt).del_z()
			else:
				loss_val,pr_gt   = self.validate(dataloader_test,gt_test,full_process=False)
			self.scheduler1.step()
			self.scheduler2.step(loss_val)
			self.writer.add_scalar('Loss/train', loss_train, epoch)
			self.writer.add_scalar('Loss/validation', loss_val, epoch)
			self.writer.add_scalar('Metrics/lr', self.scheduler1.get_last_lr()[0], epoch)
			print()
			print(f'Epoch [{epoch+1}/{self.num_epochs}] - ' f'Train Loss: {loss_train:.4f} --------- ' f'Validation Loss: {loss_val:.4f}')
			print()
			self.metric_cur = loss_val
			self.save_checkpoint(epoch)

			if (epoch+1)%5==0:
				print('epoch summary:')
				print('JI = {:.4f}'.format(ji))
				print('rmse3D = {:.4f}'.format(rmse3d))
				print('LR = {:.8f}'.format(self.scheduler1.get_last_lr()[0]))
				for _ in range(3):
					print()
				self.writer.add_scalar('Metrics/jaccardian_index',ji,epoch)
				self.writer.add_scalar('Metrics/precision',pre,epoch)
				self.writer.add_scalar('Metrics/recall',rec,epoch)
				self.writer.add_scalar('Metrics/rmse_3D',rmse3d,epoch)
				self.writer.add_scalar('Metrics/rmse_2D',rmse2d,epoch)
				self.writer.add_scalar('Metrics/rmse_x',rmsex,epoch)
				self.writer.add_scalar('Metrics/rmse_y',rmsey,epoch)
				self.writer.add_scalar('Metrics/rmse_z',rmsez,epoch)
				self.writer.add_scalar('Metrics/delx',delx,epoch)
				self.writer.add_scalar('Metrics/dely',dely,epoch)
				self.writer.add_scalar('Metrics/delz',delz,epoch)
				


if __name__ == '__main__':
	trainer = LiveTrainer(param)
	trainer.train()