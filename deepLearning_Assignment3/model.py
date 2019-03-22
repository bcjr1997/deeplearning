import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def initiate_basic_model(x):
    #Linear Architecture Model
    with tf.name_scope('linear_model') as scope:
        hidden = tf.layers.conv2d(x, 400, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='hidden_layer')
        output = tf.layers.dense(hidden, 18, name='output_layer',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    
    tf.identity(output, name='output')
    return scope, output

def initiate_better_model(x):
    #3 Layer Architecture model
    with tf.name_scope('3_layer_model') as scope:
        hidden_1 = tf.layers.conv2d(x, 400, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='output_layer')
        dropout_1 = tf.layers.dropout(hidden_1, training=True, name='dropout_layer_1')
        hidden_2 = tf.layers.conv2d(dropout_1, 250, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='output_layer')
        dropout_2 = tf.layers.dropout(hidden_2, training=True, name='dropout_layer_2')
        output = tf.layers.dense(dropout_2, 18, name='output_layer',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    
    tf.identity(output, name='output')
    return scope, output

