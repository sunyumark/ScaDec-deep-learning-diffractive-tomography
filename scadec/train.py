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
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
from scadec import util

class Trainer_bn(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    
    the phase of the unet are True by default

    """
    
    def __init__(self, net, batch_size=1, optimizer="adam", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs  

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.loss,
                                                                                global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                                   **self.opt_kwargs).minimize(self.net.loss,
                                                                                global_step=global_step)
        
        return optimizer
        
    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)
        logging.getLogger().setLevel(logging.INFO)

        # get optimizer
        self.optimizer = self._get_optimizer(training_iters, global_step)
        init = tf.global_variables_initializer()

        # get validation_path
        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        
        return init

    def train(self, data_provider, output_path, valid_provider, valid_size, training_iters=100, epochs=1000, dropout=0.75, display_step=1, save_epoch=50, restore=False, write_graph=False, prediction_path='validation'):
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param valid_provider: data provider for the validation dataset
        :param valid_size: batch size for validation provider
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        
        # initialize the training process.
        init = self._initialize(training_iters, output_path, restore, prediction_path)

        # create output path
        directory = os.path.join(output_path, "final/")
        if not os.path.exists(directory):
            os.makedirs(directory)

        save_path = os.path.join(directory, "model.cpkt")
        if epochs == 0:
            return save_path

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")

            # select validation dataset
            valid_x, valid_y = valid_provider(valid_size, fix=True)
            util.save_mat(valid_y, "%s/%s.mat"%(self.prediction_path, 'origin_y'))
            util.save_mat(valid_x, "%s/%s.mat"%(self.prediction_path, 'origin_x'))

            for epoch in range(epochs):
                total_loss = 0
                # batch_x, batch_y = data_provider(self.batch_size)
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)
                    # Run optimization op (backprop)
                    _, loss, lr, avg_psnr = sess.run([self.optimizer,
                                                        self.net.loss, 
                                                        self.learning_rate_node, 
                                                        self.net.avg_psnr], 
                                                        feed_dict={self.net.x: batch_x,
                                                                    self.net.y: batch_y,
                                                                    self.net.keep_prob: dropout,
                                                                    self.net.phase: True})
                    
                    if step % display_step == 0:
                        logging.info("Iter {:} (before training on the batch) Minibatch MSE= {:.4f}, Minibatch Avg PSNR= {:.4f}".format(step, loss, avg_psnr))
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x, batch_y)
                        
                    total_loss += loss

                    self.record_summary(summary_writer, 'training_loss', loss, step)
                    self.record_summary(summary_writer, 'training_avg_psnr', avg_psnr, step)

                # output statistics for epoch
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.output_valstats(sess, summary_writer, step, valid_x, valid_y, "epoch_%s"%epoch, store_img=True)

                if epoch % save_epoch == 0:
                    directory = os.path.join(output_path, "{}_cpkt/".format(step))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    path = os.path.join(directory, "model.cpkt".format(step))      
                    self.net.save(sess, path)

                save_path = self.net.save(sess, save_path)

            logging.info("Optimization Finished!")
            
            return save_path
    
    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info("Epoch {:}, Average MSE: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        loss, predictions, avg_psnr = sess.run([self.net.loss,  
                                                self.net.recons,
                                                self.net.avg_psnr], 
                                                feed_dict={self.net.x: batch_x,
                                                            self.net.y: batch_y,
                                                            self.net.keep_prob: 1.,
                                                            self.net.phase: False})

        self.record_summary(summary_writer, 'minibatch_loss', loss, step)
        self.record_summary(summary_writer, 'minibatch_avg_psnr', avg_psnr, step)

        logging.info("Iter {:} (After training on the batch) Minibatch MSE= {:.4f}, Minibatch Avg PSNR= {:.4f}".format(step,loss,avg_psnr))

    def output_valstats(self, sess, summary_writer, step, batch_x, batch_y, name, store_img=True):
        prediction, loss, avg_psnr = sess.run([self.net.recons,
                                                self.net.valid_loss,
                                                self.net.valid_avg_psnr], 
                                                feed_dict={self.net.x: batch_x, 
                                                            self.net.y: batch_y,
                                                            self.net.keep_prob: 1.,
                                                            self.net.phase: False})

        self.record_summary(summary_writer, 'valid_loss', loss, step)
        self.record_summary(summary_writer, 'valid_avg_psnr', avg_psnr, step)

        logging.info("Validation Statistics, validation loss= {:.4f}, Avg PSNR= {:.4f}".format(loss, avg_psnr))

        util.save_mat(prediction, "%s/%s.mat"%(self.prediction_path, name))

        if store_img:
            util.save_img(prediction[0,...], "%s/%s_img.tif"%(self.prediction_path, name))

    def record_summary(self, writer, name, value, step):
        summary=tf.Summary()
        summary.value.add(tag=name, simple_value = value)
        writer.add_summary(summary, step)
        writer.flush()

