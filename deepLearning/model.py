import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def initiate_basic_model(x, y):
    #Linear Architecture Model
    with tf.name_scope('linear_model') as scope:
        hidden = tf.layers.dense(x, 400, 
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='hidden_layer')
        output = tf.layers.dense(hidden, 10, name='output_layer',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
        tf.identity(output, name='output')
    return scope, output

def initiate_better_model(x, y):
    #3 Layer Architecture model
    with tf.name_scope('3_layer_model') as scope:
        hidden_1 = tf.layers.dense(x, 400, 
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='hidden_layer_1')
        hidden_2 = tf.layers.dense(hidden_1, 196, 
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='hidden_layer_2')
        output = tf.layers.dense(hidden_2, 10, name='output_layer',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
        tf.identity(output, name='output')
    return scope, output

