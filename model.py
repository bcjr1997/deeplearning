import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def initiate_policy_model(x, action_space):
    #Linear Architecture Model
    with tf.name_scope('policy_model') as scope:
        hidden = tf.layers.conv2d(x, 400, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='policy_hidden_layer')
        flat_layer_1 = tf.contrib.layers.flatten(hidden)
        output = tf.layers.dense(flat_layer_1, action_space,
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    tf.identity(output, name='output')
    return scope, output

def initiate_target_model(x, action_space):
    #Linear Architecture Model
    with tf.name_scope('target_model') as scope:
        hidden = tf.layers.conv2d(x, 400, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='target_hidden_layer')
        flat_layer_1 = tf.contrib.layers.flatten(hidden)
        output = tf.layers.dense(flat_layer_1, action_space,
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    tf.identity(output, name='output')
    return scope, output

def initiate_better_policy_model(x, action_space):
    #3 Layer Architecture model
    with tf.name_scope('multi_layer_policy_model') as scope:
        conv_1 = tf.layers.conv2d(x, 400, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='policy_conv_layer_1')
        conv_2 = tf.layers.conv2d(conv_1, 350, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='policy_conv_layer_2')
        conv_3 = tf.layers.conv2d(conv_2, 300, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='policy_conv_layer_3')
        conv_4 = tf.layers.conv2d(conv_3, 250, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='policy_conv_layer_4')
        fc_layer_1 = tf.layers.dense(conv_4, 200, name='policy_dense_layer_1',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
        fc_layer_2 = tf.layers.dense(fc_layer_1, 100, name='policy_dense_layer_2',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
        flat_layer_1 = tf.contrib.layers.flatten(fc_layer_2)
        output = tf.layers.dense(flat_layer_1, action_space,
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    
    tf.identity(output, name='output')
    return scope, output

def initiate_better_target_model(x, action_space):
    #3 Layer Architecture model
    with tf.name_scope('multi_layer_target_model') as scope:
        conv_1 = tf.layers.conv2d(x, 400, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='target_conv_layer_1')
        conv_2 = tf.layers.conv2d(conv_1, 350, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='target_conv_layer_2')
        conv_3 = tf.layers.conv2d(conv_2, 300, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='target_conv_layer_3')
        conv_4 = tf.layers.conv2d(conv_3, 250, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='target_conv_layer_4')
        fc_layer_1 = tf.layers.dense(conv_4, 200, name='target_dense_layer_1',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
        fc_layer_2 = tf.layers.dense(fc_layer_1, 100, name='target_dense_layer_2',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
        flat_layer_1 = tf.contrib.layers.flatten(fc_layer_2)
        output = tf.layers.dense(flat_layer_1, action_space, name='target_output_layer',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    
    tf.identity(output, name='output')
    return scope, output

