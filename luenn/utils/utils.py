import numpy as np
import pandas as pd
import torch

from luenn.model.model import UNet


def load_model(dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.to(device)
    loaded_model = torch.load(dir)
    if isinstance(loaded_model, dict):
        print(f"It's a state_dict saved at epoch {loaded_model['epoch']}")
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
    if x.ndim == 4:
        xnorm = np.zeros((x.shape[0],x.shape[1],x.shape[2]))
        for i in range(x.shape[0]):
            xnorm[i]+= (x[i,:,:,0]**2+x[i,:,:,1]**2)**.5
        return xnorm
    else:
        return (x[:,:,0]**2+x[:,:,1]**2)**.5
# import decode
# import pkg_resources
# import os
# import yaml
# def param_reference():
#     dir = pkg_resources.resource_stream(__name__,'config/param_ref.yaml').name
#     param = decode.utils.param_io.load_params(dir)
#     print(param)
#     return param

# def param_preparation(param):
#     {param.HyperParameter.lr,}
#     param = decode.utils.param_io.autoset_scaling(param)
#     return param

# def param_load(dir):
#     param = decode.utils.param_io.load_params(dir)
#     param = decode.utils.types.RecursiveNamespace.map_entry(param)
#     return param

# log_directory = "./log"
# if not os.path.exists(log_directory):
#     os.makedirs(log_directory)
# ##parameter file

# class ParamHandling:
#     class RecursiveNamespace:
#         def __init__(self, dictionary):
#             for key, value in dictionary.items():
#                 if isinstance(value, dict):
#                     setattr(self, key, ParamHandling.RecursiveNamespace(value))
#                 else:
#                     setattr(self, key, value)

#     @staticmethod
#     def param_load(load_directory):
#         with open(load_directory, 'r') as f:
#             param = yaml.safe_load(f)
#         return param

#     @staticmethod
#     def namespace_to_dict(namespace):
#         if isinstance(namespace, ParamHandling.RecursiveNamespace):
#             return {key: ParamHandling.namespace_to_dict(value) for key, value in namespace.__dict__.items() if not key.startswith("__")}
#         else:
#             return namespace

# default_params = ParamHandling.param_load('./package/config/param_defaults.yaml')
# existing_params = ParamHandling.param_load('./param/param.yaml')
# param = ParamHandling.namespace_to_dict(existing_params)

# # print(default_params)
# def fill_defaults(params, default_params):
#     for key, value in default_params.items():
#         if key not in params:
#             params[key] = value
#         elif isinstance(value, dict) and isinstance(params[key], dict):
#             fill_defaults(params[key], value)
#         elif not params[key]:
#             params[key] = value
