from Localization3D.luenn.model import Unet
import torch
import optuna
def define_model(trial):
  initializer = nn.init.kaiming_normal_
  activation = nn.ELU
  input_channels = 1
  pred_channels = trial.suggest_categorical('pred_channels', [64, 128, 256])
  kernel_size_encoder = trial.suggest_categorical('kernel_size_encoder', [3, 5, 7, 9])
  kernel_size_decoder = trial.suggest_categorical('kernel_size_decoder', [3, 5, 7, 9])
  kernel_size_output = trial.suggest_categorical('kernel_size_output', [3, 5, 7, 9])
  initializer = trial.suggest_categorical('initializer', [nn.init.kaiming_uniform_, nn.init.xavier_uniform_,  nn.init.xavier_,  nn.init.kaiming_normal_])
  activation = trial.suggest_categorical('activation', [nn.ELU,nn.GELU,nn.ReLU])
  model = UNet(initializer, activation, input_channels, pred_channels,kernel_size_encoder,kernel_size_decoder,kernel_size_output)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  return model