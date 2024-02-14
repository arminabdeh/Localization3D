import torch

class CustomLoss:
	def __init__(self, param, alpha_rate=0.01, writer=None):
		self.writer = writer
		self.param = param
		self.alpha = 0.5
		self.alpha_rate = alpha_rate
	def __call__(self, outputs, targets, gt, step=None):
		mean_channels = outputs[:, 0:2, :, :]
		var_channels = outputs[:, 2:, :, :]
		num_seeds = len(gt)
		nll_loss_log = 250*torch.log(var_channels)
		nll_loss_log = torch.clamp(nll_loss_log, max=1e12)
		nll_loss_log = torch.sum(nll_loss_log) / num_seeds
		nll_loss_exp = torch.square(mean_channels - targets) / var_channels
		nll_loss_exp = torch.clamp(nll_loss_exp, max=1e12)
		nll_loss_exp = torch.sum(nll_loss_exp) / num_seeds
		if nll_loss_exp > nll_loss_log:
			self.alpha -= self.alpha_rate
		else:
			self.alpha += self.alpha_rate

		loss = self.alpha * nll_loss_log + (1 - self.alpha) * nll_loss_exp

		mse_loss = torch.sum(torch.square(mean_channels - targets)) / num_seeds
		mean_var = torch.mean(var_channels)
		mean_mu = torch.mean(mean_channels)
		if self.writer is not None and step is not None:
			self.writer.add_scalar('Loss/epoch_loss', loss.item(), global_step=step)
			self.writer.add_scalar('Loss/mse_loss', mse_loss.item(), global_step=step)
			self.writer.add_scalar('Loss/mean_var', mean_var.item(), global_step=step)
			self.writer.add_scalar('Loss/mean_mu', mean_mu.item(), global_step=step)
			self.writer.add_scalar('Loss/nll_loss_log', nll_loss_log.item(), global_step=step)
			self.writer.add_scalar('Loss/nll_loss_exp', nll_loss_exp.item(), global_step=step)
		return loss
