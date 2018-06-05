# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Modified on Feb, 2018 based on the work of jakeret

author: yusun
'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import math
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

from scadec import util
from scadec.layers import *
from scadec.nets import *


class Unet_bn(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    :param kwargs: args passed to create_net function. 
    """
   
    def __init__(self, img_channels=3, truth_channels=3, cost="mean_squared_error", cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        # basic variables
        self.summaries = kwargs.get("summaries", True)
        self.img_channels = img_channels
        self.truth_channels = truth_channels

        # placeholders for input x and y
        self.x = tf.placeholder("float", shape=[None, None, None, img_channels])
        self.y = tf.placeholder("float", shape=[None, None, None, truth_channels])
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        # reused variables
        self.nx = tf.shape(self.x)[1]
        self.ny = tf.shape(self.x)[2]       
        self.num_examples = tf.shape(self.x)[0]            

        # variables need to be calculated
        self.recons = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
        self.loss = self._get_cost(cost, cost_kwargs)
        self.valid_loss = self._get_cost(cost, cost_kwargs)
        self.avg_psnr = self._get_measure('avg_psnr')
        self.valid_avg_psnr =  self._get_measure('avg_psnr')

    def _get_measure(self, measure):
        total_pixels = self.nx * self.ny * self.truth_channels
        dtype       = self.x.dtype
        flat_recons = tf.reshape(self.recons, [-1, total_pixels])
        flat_truths = tf.reshape(self.y, [-1, total_pixels])

        if measure == 'psnr':
            # mse are of the same length of the truths
            mse = mse_array(flat_recons, flat_truths, total_pixels)
            term1 = log(tf.constant(1, dtype), 10.)
            term2 = log(mse, 10.)
            psnr = tf.scalar_mul(20., term1) - tf.scalar_mul(10., term2)
            result = psnr

        elif measure == 'avg_psnr':
            # mse are of the same length of the truths
            mse = mse_array(flat_recons, flat_truths, total_pixels)
            term1 = log(tf.constant(1, dtype), 10.)
            term2 = log(mse, 10.)
            psnr = tf.scalar_mul(20., term1) - tf.scalar_mul(10., term2)
            avg_psnr = tf.reduce_mean(psnr)
            result = avg_psnr

        else:
            raise ValueError("Unknown measure: "%cost_name)

        return result
        
    def _get_cost(self, cost_name, cost_kwargs):
        """
        Constructs the cost function.

        """

        total_pixels = self.nx * self.ny * self.truth_channels
        flat_recons = tf.reshape(self.recons, [-1, total_pixels])
        flat_truths = tf.reshape(self.y, [-1, total_pixels])
        if cost_name == "mean_squared_error":
            loss = tf.losses.mean_squared_error(flat_recons, flat_truths)
            # the mean_squared_error is equal to the following code
            # se = tf.squared_difference(flat_recons, flat_truths)
            # loss = tf.reduce_mean(se, 1)

            # add new loss function here
        else:
            raise ValueError("Unknown cost function: "%cost_name)
            
        return loss     
    
    # predict
    def predict(self, model_path, x_test, keep_prob, phase):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            prediction = sess.run(self.recons, feed_dict={self.x: x_test, 
                                                          self.keep_prob: keep_prob, 
                                                          self.phase: phase})  # set phase to False for every prediction
                            # define operation
        return prediction
    
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)