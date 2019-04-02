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
        output = tf.contrib.layers.flatten(hidden)
        output = tf.layers.dense(output, action_space, name='policy_output_layer',
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
        output = tf.contrib.layers.flatten(hidden)
        output = tf.layers.dense(output, action_space, name='target_output_layer',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    tf.identity(output, name='output')
    return scope, output

def initiate_better_policy_model(x, action_space):
    #3 Layer Architecture model
    with tf.name_scope('3_layer_policy_model') as scope:
        hidden_1 = tf.layers.conv2d(x, 400, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='policy_output_layer1')
        dropout_1 = tf.layers.dropout(hidden_1, training=True, name='policy_dropout_layer_1')
        hidden_2 = tf.layers.conv2d(dropout_1, 250, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='policy_output_layer2')
        dropout_2 = tf.layers.dropout(hidden_2, training=True, name='dropout_layer_2')
        output = tf.contrib.layers.flatten(dropout_2)
        output = tf.layers.dense(output, action_space, name='policy_output_layer',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    
    tf.identity(output, name='output')
    return scope, output

def initiate_better_target_model(x, action_space):
    #3 Layer Architecture model
    with tf.name_scope('3_layer_target_model') as scope:
        hidden_1 = tf.layers.conv2d(x, 400, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='target_output_layer1')
        dropout_1 = tf.layers.dropout(hidden_1, training=True, name='target_dropout_layer_1')
        hidden_2 = tf.layers.conv2d(dropout_1, 250, 3, (1,1), "same",
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                    activation=tf.nn.relu, name='target_output_layer2')
        dropout_2 = tf.layers.dropout(hidden_2, training=True, name='target_dropout_layer_2')
        output = tf.contrib.layers.flatten(dropout_2)
        output = tf.layers.dense(output, action_space, name='target_output_layer',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                    )
    
    tf.identity(output, name='output')
    return scope, output

