import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def initiate_basic_model(x):
    #Linear Architecture Model
    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.relu, name='hidden_1')
        hidden_2 = tf.layers.conv2d(hidden_1, 64, 3, padding='same', activation=tf.nn.relu, name='hidden_2')
        pool_1 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same')
        dropout_1 = tf.nn.dropout(pool_1, rate=0.25, name='dropout_1')
        hidden_3 = tf.layers.conv2d(dropout_1, 128, 3, padding='same', activation=tf.nn.relu, name='hidden_3')
        hidden_4 = tf.layers.conv2d(hidden_3, 128, 2, padding='same', activation=tf.nn.relu, name='hidden_4')
        pool_2 = tf.layers.max_pooling2d(hidden_4, 2, 2, padding='same')
        dropout_2 = tf.nn.dropout(pool_2, rate=0.25, name='dropout_2')
        # followed by a dense layer output
        flat = tf.reshape(dropout_2, [-1, 8*8*64]) # flatten from 4D to 2D for dense layer
        output = tf.layers.dense(flat, 10, name='output')
    
    tf.identity(output, name='output')
    return scope, output

def initiate_better_model(x):
    #3 Layer Architecture model
    with tf.name_scope('3_layer_model') as scope:
        hidden_1 = tf.layers.dense(x, 588, 
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='hidden_layer_1')
        dropout_1 = tf.layers.dropout(hidden_1, training=True, name='dropout_layer_1')
        hidden_2 = tf.layers.dense(dropout_1, 392, 
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='hidden_layer_2')
        dropout_2 = tf.layers.dropout(hidden_2, training=True, name='dropout_layer_2')
        output = tf.layers.dense(dropout_2, 10, name='output_layer',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    
    tf.identity(output, name='output')
    return scope, output

