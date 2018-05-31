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

#import cv2
import glob
import numpy as np
from PIL import Image


class BaseDataProvider(object):

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
    
    def __call__(self, n, fix=False):
        if type(n) == int and not fix:
            # X and Y are the images and truths
            train_data, truths = self._next_batch(n)
        elif type(n) == int and fix:
            train_data, truths = self._fix_batch(n)
        elif type(n) == str and n == 'full':
            train_data, truths = self._full_batch() 
        else:
            raise ValueError("Invalid batch_size: "%n)
        
        return train_data, truths

    def _next_batch(self, n):
        pass

    def _full_batch(self):
        pass


class SimpleDataProvider(BaseDataProvider):
    
    def __init__(self, data, truths):
        super(SimpleDataProvider, self).__init__()
        self.data = np.float64(data)
        self.truths = np.float64(truths)
        self.img_channels = self.data[0].shape[2]
        self.truth_channels = self.truths[0].shape[2]
        self.file_count = data.shape[0]

    def _next_batch(self, n):
        idx = np.random.choice(self.file_count, n, replace=False)
        img = self.data[idx[0]]
        nx = img.shape[0]
        ny = img.shape[1]
        X = np.zeros((n, nx, ny, self.img_channels))
        Y = np.zeros((n, nx, ny, self.truth_channels))
        for i in range(n):
            X[i] = self._process_data(self.data[idx[i]])
            Y[i] = self._process_truths(self.truths[idx[i]])
        return X, Y

    def _fix_batch(self, n):
        # first n data
        img = self.data[0]
        nx = img.shape[0]
        ny = img.shape[1]
        X = np.zeros((n, nx, ny, self.img_channels))
        Y = np.zeros((n, nx, ny, self.truth_channels))
        for i in range(n):
            X[i] = self._process_data(self.data[i])
            Y[i] = self._process_truths(self.truths[i])
        return X, Y

    def _full_batch(self):
        return self.data, self.truths

    def _process_truths(self, truth):
        # normalization by channels
        truth = np.clip(np.fabs(truth), self.a_min, self.a_max)
        for channel in range(self.truth_channels):
            truth[:,:,channel] -= np.amin(truth[:,:,channel])
            truth[:,:,channel] /= np.amax(truth[:,:,channel])
        return truth

    def _process_data(self, data):
        # normalization by channels
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        for channel in range(self.img_channels):
            data[:,:,channel] -= np.amin(data[:,:,channel])
            data[:,:,channel] /= np.amax(data[:,:,channel])
        return data
