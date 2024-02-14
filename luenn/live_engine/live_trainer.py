import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from luenn.generic import modified_fly_simulator as fly_simulator
from luenn.live_engine.data_loader_stream import data_loader_stream
from luenn.localization import localizer_machine
from luenn.model.model import UNet
from luenn.utils import param_save, auto_scaling
from luenn.utils import visualize_results, report_performance
from luenn.live_engine.loss import CustomLoss as custom_loss


# Note
# command line to see the tensorboard:
# tensorboard --logdir=runs --port=6006 --bind_all --samples_per_plugin images=100

class live_trainer:
    def __init__(self, param, model_initial=None, dir=None, mode='train'):
        self.mode = mode
        self.param = param
        self.path = None
        self.device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device2 = torch.device(
            "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else self.device1)
        # Initialize the model
        if model_initial is not None:
            self.model = model_initial
        else:
            self.model = UNet()

        self.model.to(self.device1)

        if param.InOut.model_in:
            checkpoint = torch.load(param.InOut.model_in)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.lr = checkpoint['lr_scheduler_state_dict']['_last_lr'][0]
            self.epoch_start = checkpoint['epoch']
            self.path = os.path.dirname(param.InOut.model_in)
            self.metric_min = checkpoint['best_loss']
            self.epoch_update = checkpoint['last_update_loss']
            print('Resume training')
            print(f"It's a state_dict saved at epoch {checkpoint['epoch']}")
            print(f"Last LR: {checkpoint['lr_scheduler_state_dict']['_last_lr'][0]}")
            print(f"Best loss: {checkpoint['best_loss']}")
            print(f"Best loss happened at epoch {checkpoint['last_update_loss']}")
        else:
            self.lr = param.HyperParameter.lr
            self.epoch_start = 0
            self.metric_min = float('inf')
            self.epoch_update = 0

        # Set default path using os.makedirs
        if dir is None:
            self.path = os.path.join(os.getcwd(), 'runs', f"checkpoints_{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}")
        else:
            self.path = os.path.join(os.getcwd(), 'runs', dir)
        os.makedirs(self.path, exist_ok=True)

        # Set checkpoint and model save paths
        self.checkpoint_save_path = os.path.join(self.path, param.InOut.model_check)
        self.model_save_path = os.path.join(self.path, param.InOut.model_out)

        # Initialize a SummaryWriter for TensorBoard
        self.writer = SummaryWriter(self.path)
        self.alpha_rate = param.HyperParameter.alpha_rate
        self.custom_loss = custom_loss(param, alpha_rate=self.alpha_rate, writer=self.writer)
        # add model graph to tensorboard
        self.writer.add_graph(self.model, torch.rand(1, 1, 64, 64).to(self.device1))

        # Initialize simulator and other training-related parameters
        self.simulator = fly_simulator(param, report=False)
        self.metric_cur = 0
        self.metric_min = float('inf')
        self.batch_size = param.HyperParameter.batch_size
        self.num_epochs = param.HyperParameter.epochs
        self.gamma = param.HyperParameter.learning_rate_scheduler_param.gamma
        self.step_size = param.HyperParameter.learning_rate_scheduler_param.step_size
        self.restart_epochs = param.HyperParameter.restart_period
        self.train_size = param.HyperParameter.pseudo_ds_size
        self.test_size = param.TestSet.test_size
        self.num_workers = param.Hardware.num_worker_train
        self.accumulative_steps = param.HyperParameter.accumulative_steps
        self.pateince = param.HyperParameter.learning_rate_scheduler_param.patience
        self.reduce_rate = param.HyperParameter.learning_rate_scheduler_param.reduce_rate
        self.norm_clip = param.HyperParameter.norm_clip
        self.train_losses = []
        self.validation_losses = []

    def __del__(self):
        # Ensure that the SummaryWriter is closed when the training is finished
        if hasattr(self, 'writer'):
            self.writer.close()

    def create_optimizer_and_schedulers(self):
        beta1 = self.param.HyperParameter.optimizer_param.beta1
        beta2 = self.param.HyperParameter.optimizer_param.beta2
        weight_decay = self.param.HyperParameter.optimizer_param.weight_decay
        amsgrad = self.param.HyperParameter.optimizer_param.amsgrad
        self.optimizer = getattr(torch.optim, "AdamW")(self.model.parameters(), lr=self.lr, betas=(beta1, beta2),
                                                       eps=1e-08, weight_decay=weight_decay, amsgrad=amsgrad)

        # Create schedulers
        self.scheduler1 = lr_scheduler.StepLR(self.optimizer, self.step_size, self.gamma)
        self.scheduler2 = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.pateince, verbose=True, factor=self.reduce_rate)

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
            print(f'metric did not improve from {self.metric_min} updated at epoch {self.epoch_update + 1}')

    def data_loader(self):
        streaming_train = data_loader_stream(self.simulator, evaluate=False)
        dataloader_train = DataLoader(streaming_train, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                                      num_workers=self.num_workers, drop_last=True)

        streaming_test = data_loader_stream(self.simulator, evaluate=True)
        dataloader_test = DataLoader(streaming_test, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                                     num_workers=self.num_workers, drop_last=True)

        gt_test = streaming_test.get_all_gts()
        gt_train = streaming_train.get_all_gts()
        return dataloader_train, dataloader_test, gt_test, gt_train

    def validate(self, dataloader_test, gt_test):
        print('Validation started')
        self.model.eval()

        steps = len(dataloader_test)
        tqdm_enum = tqdm(total=steps, smoothing=0.)
        val_loss = 0
        pr_gt_total = pd.DataFrame([])
        example = None
        example_pr_gt = None

        with torch.no_grad():
            for idx, data in enumerate(dataloader_test):
                frame_id_batch_start = int(idx * self.batch_size)
                frame_id_batch_end = int(frame_id_batch_start + self.batch_size)
                gt_batch = gt_test[gt_test['frame_id'].between(frame_id_batch_start + 1, frame_id_batch_end)]

                inputs, labels = data['x'].to(self.device1), data['y'].to(self.device2)
                outputs = self.model(inputs)
                outputs = outputs.to(self.device2)

                loss = self.custom_loss(outputs, labels, gt=gt_batch)
                val_loss += loss.item()

                outputs = outputs.cpu().numpy()
                outputs = np.moveaxis(outputs, 1, -1)
                pr_gt_temp = localizer_machine(outputs, gt=gt_batch, save=False, param=self.param).localization_3D()
                pr_gt_total = pd.concat([pr_gt_total, pr_gt_temp])

                inputs = inputs.cpu()
                labels = labels.cpu()
                torch.cuda.empty_cache()
                tqdm_enum.update(1)

                if idx == 0:
                    ex_labels = labels.cpu().numpy()
                    ex_labels = np.moveaxis(ex_labels, 1, -1)
                    example = np.concatenate((outputs, ex_labels), axis=-1)
                    example_pr_gt = pr_gt_temp

        tqdm_enum.close()
        val_loss /= steps
        return val_loss, pr_gt_total, example, example_pr_gt


    def train_one_epoch(self, dataloader_train, gt_train, epochs=0, warmup=False):
        if warmup:
            print('Warmup started')
        else:
            print(f'Training epoch {epochs + 1}')
        self.model.train()
        steps = len(dataloader_train)
        tqdm_enum = tqdm(total=steps, smoothing=0.)
        train_loss = 0


        for batch_idx, data in enumerate(dataloader_train):
            global_steps = batch_idx + epochs * steps
            frame_id_batch_start = int(batch_idx * self.batch_size)
            frame_id_batch_end = int(frame_id_batch_start + self.batch_size)
            gt_batch = gt_train[gt_train['frame_id'].between(frame_id_batch_start + 1, frame_id_batch_end)]

            inputs, labels = data['x'].to(self.device1), data['y'].to(self.device2)
            outputs = self.model(inputs)
            outputs = outputs.to(self.device2)
            if warmup:
                loss = self.custom_loss(outputs, labels, gt_batch, step=None)
            else:
                loss = self.custom_loss(outputs, labels, gt_batch, step=global_steps)
            train_loss += loss.item()
            loss = loss / self.accumulative_steps
            if torch.isnan(loss):
                print('nan encountered')
                continue

            loss.backward()

            if batch_idx % self.accumulative_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.norm_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            tqdm_enum.update(1)
            inputs = inputs.cpu()
            labels = labels.cpu()
            torch.cuda.empty_cache()

        tqdm_enum.close()
        train_loss /= steps
        if warmup:
            print(f'Warmup finished with loss {train_loss:.4f} per seed')
        return train_loss

    def train(self):
        torch.cuda.empty_cache()
        filename = os.path.join(self.path, 'param_in.yaml')
        param_save(self.param, filename)
        self.create_optimizer_and_schedulers()
        dataloader_train, dataloader_test, gt_test, gt_train = self.data_loader()
        warmup_loss = self.train_one_epoch(dataloader_train, gt_train, epochs=0, warmup=True)
        print('training started')
        for epoch in range(self.epoch_start, self.num_epochs):
            if (epoch + 1) % self.restart_epochs == 0 and epoch != 0:
                print('-' * 50)
                print('restart simulation setup')
                print('-' * 50)
                self.simulator = fly_simulator(self.param, report=False)
            dataloader_train, dataloader_test, gt_test, gt_train = self.data_loader()
            loss_train = self.train_one_epoch(dataloader_train, gt_train, epochs=epoch)
            loss_val, pr_gt, temp_out, pr_gt_temp = self.validate(dataloader_test, gt_test)
            self.scheduler1.step()
            self.scheduler2.step(loss_val)
            print(f'epoch {epoch + 1} finished')
            print(f'Train Loss: {loss_train:.4f} --------- Validation Loss: {loss_val:.4f}')
            print(f'epoch summary:')
            print(f'LR = {self.scheduler1.get_last_lr()[0]:.8f}')
            rec, pre, ji, rmse3d, rmse2d, rmsez, rmsex, rmsey, delx, dely, delz = report_performance(pr_gt)
            self.writer.add_scalar('Loss/val', loss_val, epoch)
            self.writer.add_scalar('Loss/train', loss_train, epoch)
            self.writer.add_scalar('Performance/recall', rec, epoch)
            self.writer.add_scalar('Performance/precision', pre, epoch)
            self.writer.add_scalar('Performance/jaccard', ji, epoch)
            self.writer.add_scalar('Performance/rmse3d', rmse3d/np.sqrt(3), epoch)
            self.writer.add_scalar('Performance/rmse2d', rmse2d/np.sqrt(2), epoch)
            self.writer.add_scalar('Performance/rmsez', rmsez, epoch)
            self.writer.add_scalar('Performance/rmsex', rmsex, epoch)
            self.writer.add_scalar('Performance/rmsey', rmsey, epoch)
            self.writer.add_scalar('Performance/delx', delx, epoch)
            self.writer.add_scalar('Performance/dely', dely, epoch)
            self.writer.add_scalar('Performance/delz', delz, epoch)
            self.writer.add_scalar('Performance/lr', self.scheduler1.get_last_lr()[0], epoch)
            fig = visualize_results(pr_gt_temp, temp_out)
            self.writer.add_figure('predictions', fig, epoch)
            self.train_losses.append(loss_train)
            self.validation_losses.append(loss_val)
            if self.mode == 'train':
                self.metric_cur = loss_val
                self.save_checkpoint(epoch)
        return self.model, self.train_losses, self.validation_losses

if __name__ == '__main__':
    from luenn.utils import param_reference
    param = param_reference()
    param.Simulation.scale_factor = 1000.0
    param.Simulation.emitter_av = 10
    param.HyperParameter.pseudo_ds_size = 512
    param.HyperParameter.batch_size = 2
    param.HyperParameter.epochs = 1000
    param.HyperParameter.restart_period = 20
    param.TestSet.test_size = 128
    param.Hardware.num_worker_train = 4
    param.HyperParameter.lr = 0.00080
    param.Simulation.label_slide = True
    param.Simulation.intensity_mu_sig = [20000, 1000]
    param = auto_scaling(param)
    live_trainer(param,dir='hsnr_ld3',mode='debug').train()
