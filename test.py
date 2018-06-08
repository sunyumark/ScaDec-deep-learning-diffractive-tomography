from scadec.unet_bn import Unet_bn
from scadec.train import Trainer_bn

from scadec import image_util
from scadec import util

import scipy.io as spio
import numpy as np
import os

####################################################
####             PREPARE WORKSPACE               ###
####################################################

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_vis = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_vis; # 0,1,2,3

# here specify the path of the model you want to load
gpu_ind = '0'
model_path = 'gpu' + gpu_ind + '/models/60099_cpkt/models/final/model.cpkt'

data_channels = 2
truth_channels = 1

####################################################
####                 FUNCTIONS                   ###
####################################################

# make the data a 4D vector
def preprocess(data, channels):
	nx = data.shape[1]
	ny = data.shape[2]
	return data.reshape((-1, nx, ny, channels))

####################################################
####                lOAD MODEL                   ###
####################################################

# set up args for the unet, should be exactly the same as the loading model
kwargs = {
    "layers": 5,
    "conv_times": 2,
    "features_root": 64,
    "filter_size": 3,
    "pool_size": 2,
    "summaries": True
}

net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost="mean_squared_error", **kwargs)


####################################################
####                lOAD TRAIN                   ###
####################################################

#preparing training data
data_mat = spio.loadmat('train_np/obhatGausWeak128_40.mat', squeeze_me=True)
truths_mat = spio.loadmat('train_np/obGausWeak128_40.mat', squeeze_me=True)

data = data_mat['obhatGausWeak128']
data = preprocess(data, data_channels) # 4 dimension -> 3 dimension if you do data[:,:,:,1]
truths = preprocess(truths_mat['obGausWeak128'], truth_channels)

data_provider = image_util.SimpleDataProvider(data, truths)


####################################################
####                 lOAD TEST                   ###
####################################################

vdata_mat = spio.loadmat('test_np_noise/obhatGausWeak{}Noise128.mat'.format(level), squeeze_me=True)
vtruths_mat = spio.loadmat('valid_np/obGausN1S128val.mat', squeeze_me=True)

vdata = vdata_mat['obhatGausWeak128']
vdata = preprocess(vdata, data_channels)
vtruths = preprocess(vtruths_mat['obGausN1S128val'], truth_channels)

valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

####################################################
####              	  PREDICT                    ###
####################################################

predicts = []

valid_x, valid_y = valid_provider('full')
num = valid_x.shape[0]

for i in range(num):

    print('')
    print('')
    print('************* {} *************'.format(i))
    print('')
    print('')

    x_train, y_train = data_provider(23)
    x_input = valid_x[i:i+1,:,:,:]
    x_input = np.concatenate((x_input, x_train), axis=0)
    predict = net.predict(model_path, x_input, 1, True)
    predicts.append(predict[0:1,:,:])

predicts = np.concatenate(predicts, axis=0)
util.save_mat(predicts, 'test{}Noise.mat'.format(level))






