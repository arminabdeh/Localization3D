import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from luenn.evaluate import reg_classification
from luenn.generic import modified_fly_simulator as fly_simulator
from luenn.localization import localizer_machine
from luenn.model.model import UNet
from luenn.utils import param_save


class training_stream(Dataset):
    def __init__(self, simulator):
        self.data = simulator.ds_train()
        self.num_frames = self.data[0].numpy().shape[0]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        x_sim, y_sim, gt_sim = self.data
        x_sample = x_sim[index]
        y_sample = y_sim[index]
        return {'x': x_sample, 'y': y_sample}

class live_trainer:
    def __init__(self, param):
        self.param = param
        self.path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet()
        self.model.to(self.device)
        if param.InOut.model_in:
            loaded_model = torch.load(param.InOut.model_in)
            checkpoint = loaded_model['model_state_dict']
            resume_training = self.param.InOut.resume_training
            self.model.load_state_dict(checkpoint)
            if resume_training:
                self.lr = loaded_model['lr_scheduler_state_dict']['_last_lr'][0]
                self.epoch_start = loaded_model['epoch']
                self.path = os.path.dirname(param.InOut.model_in)
                self.metric_min = loaded_model['best_loss']
                self.epoch_update = loaded_model['last_update_loss']
                print('resume training')
                print(f"It's a state_dict saved at epoch {loaded_model['epoch']}")
                print(f" last lr {loaded_model['lr_scheduler_state_dict']['_last_lr'][0]}")
                print(f" best loss {loaded_model['best_loss']}")
                print(f"best loss happened at epoch {loaded_model['last_update_loss']}")
            else:
                self.lr = param.HyperParameter.lr
                self.epoch_start = 0
                self.path = None
                self.model.to(self.device)
                self.metric_min = float('inf')
                self.epoch_update = 0
        else:
            self.lr = param.HyperParameter.lr
            self.epoch_start = 0
            self.model.to(self.device)
            self.metric_min = float('inf')
            self.epoch_update = 0
        if self.path is None:
            path = os.path.join(os.getcwd(), 'runs')
            if not os.path.exists(path):
                os.mkdir(path)
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            folder_name = f"checkpoints_{timestamp}"
            path = os.path.join(path, folder_name)
            if not os.path.exists(path):
                os.mkdir(path)
            self.path = path
        self.checkpoint_save_path = os.path.join(self.path, param.InOut.model_check)
        self.model_save_path = os.path.join(self.path, param.InOut.model_out)
        self.writer = SummaryWriter(self.path)
        self.writer.add_graph(self.model, torch.zeros(1, 1, 64, 64).to(self.device))
        filename = os.path.join(self.path, 'param_in.yaml')
        param_save(param, filename)
        ### training parameters
        self.simulator = fly_simulator(param, report=False)
        self.metric_cur = 0
        self.batch_size = param.HyperParameter.batch_size
        self.accumulation_steps = 64 // self.batch_size
        self.num_epochs = param.HyperParameter.epochs
        self.gamma = param.HyperParameter.learning_rate_scheduler_param.gamma
        self.step_size = param.HyperParameter.learning_rate_scheduler_param.step_size
        self.restart_epochs = param.HyperParameter.restart_period
        self.val_localization_period = param.HyperParameter.val_localization_period
        self.train_size = param.HyperParameter.pseudo_ds_size
        self.test_size = param.TestSet.test_size
        self.num_workers = self.param.Hardware.num_worker_train
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.validation_losses = []
        beta1 = param.HyperParameter.optimizer_param.beta1
        beta2 = param.HyperParameter.optimizer_param.beta2
        weight_decay = param.HyperParameter.optimizer_param.weight_decay
        amsgrad = param.HyperParameter.optimizer_param.amsgrad
        self.optimizer = getattr(torch.optim, "AdamW")(self.model.parameters(), lr=self.lr, betas=(beta1, beta2),
                                                       eps=1e-08, weight_decay=weight_decay, amsgrad=amsgrad)
        self.scheduler1 = lr_scheduler.StepLR(self.optimizer, self.step_size, gamma=self.gamma)
        self.scheduler2 = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=20, verbose=True, factor=0.5)
    def train_one_epoch(self, dataloader_train, accumulation_steps=1):
        self.model.train()
        steps = len(dataloader_train)
        tqdm_enum = tqdm(total=int(steps / accumulation_steps) + 1, smoothing=0.)
        train_loss = 0
        for batch_idx, data in enumerate(dataloader_train):
            inputs, labels = data['x'], data['y']
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            train_loss += loss.item()
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(dataloader_train) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                tqdm_enum.update(1)
            inputs.to('cpu')
            labels.to('cpu')
            torch.cuda.empty_cache()
        return train_loss / len(dataloader_train)

    def save_checkpoint(self, epoch_cur):
        print('*' * 50)
        torch.save({'epoch': epoch_cur, 'last_update_loss': self.epoch_update + 1, 'best_loss': self.metric_min,
                    'model_state_dict': self.model.state_dict(),
                    'lr_scheduler_state_dict': self.scheduler1.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, self.checkpoint_save_path)
        if self.metric_cur < self.metric_min:
            print(f'metric improved from {self.metric_min} to {self.metric_cur}')
            torch.save({'epoch': epoch_cur, 'model_state_dict': self.model.state_dict(),
                        'lr_scheduler_state_dict': self.scheduler1.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, self.model_save_path)
            self.metric_min = self.metric_cur
            self.epoch_update = epoch_cur
        else:
            print(f'current metric is {self.metric_cur}')
            print(f'metric did not improve from {self.metric_min} updated at epoch {self.epoch_update + 1}')
        print('*' * 50)
        print()

    def validate(self, dataloader_test, gt_val, full_process=False):
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
                inputs.cpu()
                labels.cpu()
                torch.cuda.empty_cache()
                if full_process:
                    temp_out = np.moveaxis(outputs.cpu().numpy(), 1, -1)
                    loc_file = temp_out if loc_file is None else np.concatenate((loc_file, temp_out), axis=0)
                tqdm_enum.update(1)
        val_loss /= len(dataloader_test)
        if full_process:
            pr_gt = localizer_machine(self.param, loc_file, gt_val, save=False).localization_3D()
        return val_loss, pr_gt

    def data_loader(self):
        streaming_dataset = training_stream(self.simulator)
        dataloader_train = DataLoader(streaming_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        x_test, y_test, gt_test = self.simulator.ds_test()
        dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True,num_workers=self.num_workers)
        return dataloader_train, dataloader_test, gt_test
    def train(self):
        for epoch in range(self.epoch_start, self.num_epochs):
            if (epoch+1) % self.restart_epochs and epoch != 0:
                print('#' * 50)
                print()
                print('restart simulation setup')
                print()
                print('#' * 50)
                self.simulator = fly_simulator(self.param, report=False)
            dataloader_train, dataloader_test, gt_test = self.data_loader()
            loss_train = self.train_one_epoch(dataloader_train, accumulation_steps=self.accumulation_steps)
            if epoch % self.val_localization_period == 0:
                loss_val, pr_gt = self.validate(dataloader_test, gt_test, full_process=True)
                ji = reg_classification(pr_gt).jaccardian_index()
                rec = reg_classification(pr_gt).recall()
                pre = reg_classification(pr_gt).precision()
                rmse3d = reg_classification(pr_gt).rmse_3d()/np.sqrt(3)
                rmse2d = reg_classification(pr_gt).rmse_2d()/np.sqrt(2)
                rmsex = reg_classification(pr_gt).rmse_x()
                rmsey = reg_classification(pr_gt).rmse_y()
                rmsez = reg_classification(pr_gt).rmse_z()
                delx = reg_classification(pr_gt).del_x()
                dely = reg_classification(pr_gt).del_y()
                delz = reg_classification(pr_gt).del_z()
                print(f'Epoch [{epoch + 1}/{self.num_epochs}] - ' f'Train Loss: {loss_train:.4f} --------- ' f'Validation Loss: {loss_val:.4f}')
                print('epoch summary:')
                print('JI = {:.4f}'.format(ji))
                print('rmse3D = {:.4f}'.format(rmse3d))
                print('LR = {:.8f}'.format(self.scheduler1.get_last_lr()[0]))
                self.writer.add_scalar('Metrics/jaccardian_index', ji, epoch)
                self.writer.add_scalar('Metrics/precision', pre, epoch)
                self.writer.add_scalar('Metrics/recall', rec, epoch)
                self.writer.add_scalar('Metrics/rmse_3D', rmse3d, epoch)
                self.writer.add_scalar('Metrics/rmse_2D', rmse2d, epoch)
                self.writer.add_scalar('Metrics/rmse_x', rmsex, epoch)
                self.writer.add_scalar('Metrics/rmse_y', rmsey, epoch)
                self.writer.add_scalar('Metrics/rmse_z', rmsez, epoch)
                self.writer.add_scalar('Metrics/delx', delx, epoch)
                self.writer.add_scalar('Metrics/dely', dely, epoch)
                self.writer.add_scalar('Metrics/delz', delz, epoch)
            else:
                loss_val, pr_gt = self.validate(dataloader_test, gt_test, full_process=False)
                print(f'Epoch [{epoch + 1}/{self.num_epochs}] - ' f'Train Loss: {loss_train:.4f} --------- ' f'Validation Loss: {loss_val:.4f}')
            self.scheduler1.step()
            self.scheduler2.step(loss_val)
            self.writer.add_scalar('Loss/train', loss_train, epoch)
            self.writer.add_scalar('Loss/validation', loss_val, epoch)
            self.writer.add_scalar('Metrics/lr', self.scheduler1.get_last_lr()[0], epoch)
            self.metric_cur = loss_val
            self.save_checkpoint(epoch)


if __name__ == '__main__':
    from luenn.utils import param_reference,auto_scaling
    param = param_reference()
    param.HyperParameter.pseudo_ds_size = 16
    param.HyperParameter.batch_size = 4
    param.HyperParameter.epochs = 20
    param.HyperParameter.restart_period = 2
    param.TestSet.test_size = 4
    param.Hardware.num_worker_train = 2
    param.Simulation.intensity_mu_sig = [1000, 10]
    param = auto_scaling(param)
    live_trainer(param).train()
