import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import util

def initiate_basic_model(x):
    #Basic Convolutional Neural Network
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.conv2d(x, 32, 5, padding='same', activation=tf.nn.relu,
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    name='hidden_layer_1')
        dropout_1 = tf.layers.dropout(hidden_1, rate = 0.1, training=True, name='dropout_layer_1')
        pool_1 = tf.layers.max_pooling2d(dropout_1, 2, 2, padding='same')
        hidden_2 = tf.layers.conv2d(pool_1, 64, 5, padding='same', activation=tf.nn.relu,
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    name='hidden_layer_2')
        dropout_2 = tf.layers.dropout(hidden_2, rate = 0.1, training=True, name='dropout_layer_2')
        pool_2 = tf.layers.max_pooling2d(dropout_2, 2, 2, padding='same')
        #dense layer output
        flat = tf.reshape(pool_2, [-1, 8*8*64])
        output = tf.layers.dense(flat, 100, name = 'output')
    tf.identity(output, name='output')
    return scope, output

def initiate_autoencoder(x, code_size):
    #Autoencoder with basic convolutional neural network
    #Placeholder value = [?, 32, 32, 3] with scale of 2
    encoder_16 = util.downscale_image(x)
    encoder_8 = util.downscale_image(encoder_16)
    encoder_4 = util.downscale_image(encoder_8)
    print(encoder_4)
    flatten_dim = np.prod(encoder_4.get_shape().as_list()[1:])
    flat = tf.reshape(tf.stop_gradient(encoder_4), [-1, flatten_dim])
    code = tf.layers.dense(flat, code_size, activation=tf.nn.relu)
    hidden_decoder = tf.layers.dense(code, 16, activation=tf.nn.relu)
    decoder_4 = tf.reshape(hidden_decoder, [-1,4,4,1])
    decoder_8 = util.upscale_image(decoder_4)
    decoder_16 = util.upscale_image(decoder_8)
    output = util.upscale_image(decoder_16)
    return code, output

