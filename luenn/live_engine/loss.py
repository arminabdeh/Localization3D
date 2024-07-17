import matplotlib.pyplot as plt
import torch
from luenn.localization import localizer_machine
from luenn.evaluate import reg_classification
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent
def gmm_loss(self,predictions, target, pr_gt):
    device = predictions.device
    pr_gt_detected = pr_gt.copy()
    pr_gt_detected = pr_gt_detected.fillna(0)
    pr_gt_detected = pr_gt_detected[pr_gt_detected['label'] == 'TP']
    xyz_var = predictions[:, 3:, :, :]
    xyz_prob = predictions[:, 0, :, :]
    xyz_tr = target[:, 3:, :, :]
    xyz_pr = torch.zeros_like(xyz_var)
    x = pr_gt_detected['X_pr_px'].tolist()
    y = pr_gt_detected['Y_pr_px'].tolist()
    z = pr_gt_detected['Z_pr_nm'].tolist()
    f = pr_gt_detected['frame_id'].tolist()
    i = pr_gt_detected['i_pr'].tolist()
    j = pr_gt_detected['j_pr'].tolist()
    for s in range(len(x)):
        xyz_pr[:, 0, int(i[s]), int(j[s])] = x[s]
        xyz_pr[:, 1, int(i[s]), int(j[s])] = y[s]
        xyz_pr[:, 2, int(i[s]), int(j[s])] = z[s]
        # prob non zero
        # 2nd channel to last
        xyz_tr = xyz_tr.permute(0, 2, 3, 1)
        xyz_pr = xyz_pr.permute(0, 2, 3, 1)
        xyz_var = xyz_var.permute(0, 2, 3, 1)
        xyz_prob = torch.where(xyz_prob > 0.7, xyz_prob, torch.zeros_like(xyz_prob))
        xyz_prob_index = torch.nonzero(xyz_prob)
        xyz_tr = xyz_tr[xyz_prob_index[:, 0], xyz_prob_index[:, 1], xyz_prob_index[:, 2], :]
        xyz_pr = xyz_pr[xyz_prob_index[:, 0], xyz_prob_index[:, 1], xyz_prob_index[:, 2], :]
        xyz_var = xyz_var[xyz_prob_index[:, 0], xyz_prob_index[:, 1], xyz_prob_index[:, 2], :]
        weight = xyz_prob[xyz_prob_index[:, 0], xyz_prob_index[:, 1], xyz_prob_index[:, 2]]
        print(weight.shape, xyz_tr.shape, xyz_pr.shape, xyz_var.shape)
        batch_size = xyz_tr.shape[0]
        # flatten
        weight = weight.reshape(batch_size, -1)
        xyz_tr = xyz_tr.reshape(batch_size, -1, 3)
        xyz_pr = xyz_pr.reshape(batch_size, -1, 3)
        xyz_var = xyz_var.reshape(batch_size, -1, 3)
        mix = Categorical(weight)
        comp = Independent(Normal(xyz_pr, xyz_var), 1)
        gmm = MixtureSameFamily(mix, comp)
        neg_log_likelihood = -gmm.log_prob(xyz_tr)
        gmm_loss = torch.mean(neg_log_likelihood)
        return gmm_loss

class CustomLoss:
    def __init__(self, writer=None):
        self.writer = writer
    def mse_loss(self, pred,targ,num_seeds):
        num_frames = pred.shape[0]
        mse_loss = torch.nn.MSELoss()(pred[:, 0:2, :, :], targ[:, 0:2, :, :])
        weight = num_frames*256*256/(1000.0*num_seeds)
        return mse_loss*weight
    def __call__(self, predictions=None, targets=None, num_seeds=None, step=None):
        loss = self.mse_loss(predictions, targets, num_seeds)
        mean_variables_ch1 = torch.mean(predictions[:, 0, :, :])
        mean_variables_ch2 = torch.mean(predictions[:, 1, :, :])
        if self.writer is not None and step is not None:
            self.writer.add_scalar('Loss/epoch_loss', loss.item(), global_step=step)
            self.writer.add_scalar('Loss/mean_variables_cos', mean_variables_ch1.item(), global_step=step)
            self.writer.add_scalar('Loss/mean_variables_sin', mean_variables_ch2.item(), global_step=step)
            # self.writer.add_scalar('Loss/mse_loss', mse_loss.item(), global_step=step)
            # self.writer.add_scalar('Loss/loss_nll', nll_loss.item(), global_step=step)
            # self.writer.add_scalar('Loss/mean_variables_varx', mean_variables_ch3.item(), global_step=step)
            # self.writer.add_scalar('Loss/mean_variables_vary', mean_variables_ch4.item(), global_step=step)
            # self.writer.add_scalar('Loss/mean_variables_varz', mean_variables_ch5.item(), global_step=step)
            # self.writer.add_scalar('Loss/mse_over_nll',mse_loss.item()/(nll_loss.item()+1e-6), global_step=step)
        return loss

if __name__ == "__main__":
    import pandas as pd
    def gmm_loss(device, pr_gt):
        pr_gt_detected = pr_gt.copy()
        pr_gt_detected['X_tr_nm'] = [x if l == 'TP' or 'FN' else y for x, y, l in zip(pr_gt_detected['X_tr_nm'], pr_gt_detected['X_pr_nm']-250.0, pr_gt_detected['label'])]
        pr_gt_detected['Y_tr_nm'] = [x if l == 'TP' or 'FN' else y for x, y, l in zip(pr_gt_detected['Y_tr_nm'], pr_gt_detected['Y_pr_nm']-250.0, pr_gt_detected['label'])]
        pr_gt_detected['Z_tr_nm'] = [x if l == 'TP' or 'FN' else y for x, y, l in zip(pr_gt_detected['Z_tr_nm'], pr_gt_detected['Z_pr_nm']-500.0, pr_gt_detected['label'])]
        pr_gt_detected = pr_gt_detected[pr_gt_detected['label'] != 'FN']
        pr_gt_detected['prob_max'] = pr_gt_detected['prob_max'] / pr_gt_detected['prob_max'].sum()
        xyz_true = torch.tensor(pr_gt_detected[['X_tr_nm', 'Y_tr_nm', 'Z_tr_nm']].values, dtype=torch.float64).to(device) # Nx3
        xyz_pred = torch.tensor(pr_gt_detected[['X_pr_nm', 'Y_pr_nm', 'Z_pr_nm']].values, dtype=torch.float64).to(device) # Nx3
        xyz_var = torch.tensor(pr_gt_detected[['X_var', 'Y_var', 'Z_var']].values, dtype=torch.float64).to(device) # Nx3
        weight = torch.tensor(pr_gt_detected['prob_max'].values, dtype=torch.float64).to(device)
        mix = Categorical(weight)
        comp = Independent(Normal(xyz_pred, torch.sqrt(xyz_var)), 1)
        gmm = MixtureSameFamily(mix, comp)
        neg_log_likelihood = -gmm.log_prob(xyz_true)
        return torch.mean(neg_log_likelihood)
    data = {
        'frame_id': [0, 1, 2, 3, 4,5],
        'label': ['TP', 'TP', 'FP', 'TP', 'TP','FN'],
        'prob_max': [1.0, 0.85, 0.4, 0.6, 0.95, np.NAN],
        'X_tr_nm': [120.0, 3232, 3340, 443.0, 523.0, 10.0],
        'Y_tr_nm': [150.0, 2500.0, 3450, 450.0, 550.0, 20.0],
        'Z_tr_nm': [-400.0, -500.0, -600, 700.0, 40.0, 30.0],
        'X_pr_nm': [120.0, 2340.0, 3300, 440.0, 520.0, np.NAN],
        'Y_pr_nm': [149.0, 2350.0, 3400, 450.0, 550.0, np.NAN],
        'Z_pr_nm': [-400.0, -510.0, -610, 710.0, 50.0, np.NAN],
        'X_var': [10, 0.2, 0.3, 0.4, 0.5, np.NAN],
        'Y_var': [23.0, 0.2, 0.3, 0.4, 0.5, np.NAN],
        'Z_var': [231.0, 0.2, 0.3, 0.4, 0.5, np.NAN],
    }
    pr_gt = pd.DataFrame(data)
    gmm = gmm_loss(torch.device('cpu'), pr_gt)
    print(gmm.item())
    # exit()