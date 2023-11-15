import decode.utils.param_io as param_io
param = param_io.load_params('./param/param.yaml')
param.HyperParameter.pseudo_ds_size = 2
param.TestSet.test_size= 2

# from luenn.generic import fly_simulator
# x,y,gt = fly_simulator(param,report=True).ds_train()
# print(x.shape,y.shape,gt.shape)
# x,y,gt = fly_simulator(param,report=True).ds_test()
# print(x.shape,y.shape,gt.shape)

from luenn.live_engine import live_trainer
live_trainer(param).train()