import numpy as np
import pandas as pd
import torch
from luenn.model.model import UNet
from argparse import Namespace
import yaml
import os
import torch.nn as nn
from datetime import datetime
def convert_to_recursive_namespace(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_to_recursive_namespace(value)
    return Namespace(**dictionary)

def param_load(load_directory):
    with open(load_directory, 'r') as f:
        param_dict = yaml.safe_load(f)
    # Convert the loaded dictionary to a RecursiveNamespace
    param_recursive_namespace = convert_to_recursive_namespace(param_dict)
    return param_recursive_namespace

def param_save(param, filename):
    if filename.endswith('.yaml'):
        with open(filename, 'w') as yaml_file:
            yaml.dump(param, yaml_file)
    else:
        raise ValueError('Filename must end with .yaml')

def generate_unique_filename(f, prefix="", extension=""):
    timestamp = datetime.now().strftime("%Y.%m.%d")
    unique_filename = f"{prefix}_{timestamp}_{str(f)}xframe{extension}"
    return unique_filename

def load_model(dir_model,param=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if param:
        initializer_str = param.architecture.initializer
        initializer_str = initializer_str.lower()
        if initializer_str == 'kaiming_uniform':
            initializer = nn.init.kaiming_uniform_
        elif initializer_str == 'xavier_uniform':
            initializer = nn.init.xavier_uniform_
        elif initializer_str == 'xavier_normal':
            initializer = nn.init.xavier_normal_
        elif initializer_str == 'kaiming_normal':
            initializer = nn.init.kaiming_normal_
        else:
            raise ValueError('initializer not recognized, choose from kaiming_uniform, xavier_uniform, xavier_normal, kaiming_normal')
        activation_str = param.architecture.activation
        activation_str = activation_str.upper()
        if activation_str == 'GELU':
            activation = nn.GELU
        elif activation_str == 'ELU':
            activation = nn.ELU
        elif activation_str == 'RELU':
            activation = nn.ReLU
        else:
            raise ValueError('activation not recognized, choose from GELU, ELU, RELU')

        pred_channels = param.architecture.pred_channels
        kernel_unet = param.architecture.kernel_unet
        kernel_HR = param.architecture.kernel_HR
        kernel_output = param.architecture.kernel_output
        input_channels = param.architecture.input_channels
        model = UNet(initializer=initializer, activation=activation, input_channels=input_channels,
                     pred_channels=pred_channels, kernel_unet=kernel_unet, kernel_HR=kernel_HR,
                     kernel_output=kernel_output)
    else:
        model = UNet()
    model.to(device)
    loaded_model = torch.load(dir_model)
    if isinstance(loaded_model, dict):
        print(
            f"It's a state_dict saved at epoch {loaded_model['epoch']} with lr {loaded_model['lr_scheduler_state_dict']}")
        checkpoint = loaded_model['model_state_dict']
    else:
        checkpoint = loaded_model.state_dict()
    model.load_state_dict(checkpoint)
    return model


def dec_luenn_gt_transform(tar_em):
    xyz = tar_em.xyz_px.numpy().tolist()
    frame_id = tar_em.frame_ix.numpy().tolist()
    photons = tar_em.phot.numpy().tolist()
    GT = pd.DataFrame({'xyz': xyz, 'frame_id': frame_id, 'photons': photons})

    GT_list = []

    for f in range(GT['frame_id'].max() + 1):
        frame_gt = GT[GT['frame_id'] == f]

        for nn, (xyz_data, photon) in enumerate(
                zip(frame_gt['xyz'], frame_gt['photons'])):
            GT_frame = {
                'frame_id': f + 1,
                'seed_id': nn + 1,
                'X_tr_px': xyz_data[0],
                'Y_tr_px': xyz_data[1],
                'X_tr_nm': xyz_data[0] * 100.0,
                'Y_tr_nm': xyz_data[1] * 100.0,
                'Z_tr_nm': xyz_data[2],
                'photons': photon
            }
            GT_list.append(GT_frame)

    GT_Frames = pd.DataFrame(GT_list)
    return GT_Frames


def complex_real_map(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
        if x.shape[-1]==256:
            x = np.moveaxis(x, 1, -1)

    if x.ndim == 4:
        xnorm = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
        for i in range(x.shape[0]):
            xnorm[i] += (x[i, :, :, 0] ** 2 + x[i, :, :, 1] ** 2) ** .5
        return xnorm
    else:
        return (x[:, :, 0] ** 2 + x[:, :, 1] ** 2) ** .5
# import decode

def param_reference():
    dir = os.path.dirname(os.path.dirname(__file__))
    param_path = os.path.join(dir,'config/param/', 'param_reference.yaml')
    param_ref = param_load(param_path)
    calib_path = os.path.join(dir,'config/calib/', 'spline_calibration_3d_as_3dcal.mat')
    param_ref.InOut.calibration_file = calib_path
    return param_ref

if __name__ == '__main__':
    print(param_reference())

