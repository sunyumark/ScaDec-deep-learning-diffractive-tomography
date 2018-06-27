from scadec.unet_bn import Unet_bn
from scadec.train import Trainer_bn

from scadec import image_util

import scipy.io as spio
import numpy as np
import os

####################################################
####                 FUNCTIONS                   ###
####################################################

# make the data a 4D vector
def preprocess(data, channels):
	nx = data.shape[1]
	ny = data.shape[2]
	return data.reshape((-1, nx, ny, channels))

####################################################
####              HYPER-PARAMETERS               ###
####################################################

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3


# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 2
truth_channels = 1

####################################################
####                DATA LOADING                 ###
####################################################

"""
	here loads all the data we need for training and validating.

"""

#-- Training Data --#
data_mat = spio.loadmat('train_np/obhatGausWeak128_40.mat', squeeze_me=True)
truths_mat = spio.loadmat('train_np/obGausWeak128_40.mat', squeeze_me=True)

data = data_mat['obhatGausWeak128']
data = preprocess(data, data_channels)    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
truths = preprocess(truths_mat['obGausWeak128'], truth_channels)

data_provider = image_util.SimpleDataProvider(data, truths)

#-- Validating Data --#
vdata_mat = spio.loadmat('valid_np/obhatGausWeak128val_40.mat', squeeze_me=True)
vtruths_mat = spio.loadmat('valid_np/obGausWeak128val_40.mat', squeeze_me=True)

vdata = vdata_mat['obhatGausWeak128val']
vdata = preprocess(vdata, data_channels)
vtruths = preprocess(vtruths_mat['obGausWeak128val'], truth_channels)

valid_provider = image_util.SimpleDataProvider(vdata, vtruths)


####################################################
####                  NETWORK                    ###
####################################################

"""
	here we specify the neural network.

"""

#-- Network Setup --#
# set up args for the unet
kwargs = {
	"layers": 5,           # how many resolution levels we want to have
	"conv_times": 2,       # how many times we want to convolve in each level
	"features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
	"filter_size": 3,      # filter size used in convolution
	"pool_size": 2,        # pooling size used in max-pooling
	"summaries": True
}

net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost="mean_squared_error", **kwargs)


####################################################
####                 TRAINING                    ###
####################################################

# args for training
batch_size = 24  # batch size for training
valid_size = 24  # batch size for validating
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

# output paths for results
output_path = 'gpu' + gpu_ind + '/models'
prediction_path = 'gpu' + gpu_ind + '/validation'
# restore_path = 'gpu001/models/50099_cpkt'

# optional args
opt_kwargs = {
		'learning_rate': 0.001
}

# make a trainer for scadec
trainer = Trainer_bn(net, batch_size=batch_size, optimizer = "adam", opt_kwargs=opt_kwargs)
path = trainer.train(data_provider, output_path, valid_provider, valid_size, training_iters=100, epochs=1000, display_step=20, save_epoch=100, prediction_path=prediction_path)





