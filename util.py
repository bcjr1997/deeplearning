import collections
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import atari_wrappers
import itertools

Transition = collections.namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_action', 'next_reward', 'following_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Huber Loss : Loss function for DQN RL
def huber_loss(loss, delta=1.0):
	return tf.where(
        tf.abs(loss) < delta,
        tf.square(loss) * 0.5,
        delta * (tf.abs(loss) - 0.5 * delta)
    )

def batch_sampling(replay_memory, batch_size):
	transistions = replay_memory.sample(batch_size) #Get training data with a batch size
	batch = Transition(*zip(*transistions))
	next_state_batch = np.array(batch.next_state, dtype=np.float32)
	state_batch = np.array(batch.state, dtype=np.float32)
	action_batch = np.array(batch.action, dtype=np.int64)
	reward_batch = np.array(batch.reward)
	following_state_batch = np.array(batch.following_state, dtype=np.float32)
	next_action_batch = np.array(batch.next_action, dtype=np.int64) # Shape (32,1)
	next_reward_batch = np.array(batch.next_reward)
	return state_batch, action_batch, reward_batch, next_state_batch, next_action_batch, next_reward_batch, following_state_batch

# Calculate the gradients for DQN
def dqn_gradient_calculation(action_batch, next_state_batch, following_state_batch, reward_batch, next_reward_batch, x, y, policy_output_layer, target_output_layer, sess, batch_size, optimizer, gamma=0.99, grad_norm_clipping=1.0):
	action_index = np.stack([np.arange(batch_size, dtype=np.int32), action_batch], axis=1)
	policy = sess.run(policy_output_layer, feed_dict={x: next_state_batch})
	state_action_values = tf.gather_nd(policy, action_index)
	target = sess.run(target_output_layer, feed_dict={x: following_state_batch})
	next_state_values = tf.reduce_max(target, axis = 1)
	expected_state_action_values = reward_batch + (gamma * next_reward_batch) + ((gamma*gamma)* next_state_values)
	td_error = state_action_values - expected_state_action_values
	curr_loss = huber_loss(td_error)
	gradients, variables = zip(*optimizer.compute_gradients(curr_loss))
	return gradients, variables, curr_loss


# Method that exploits the system if the probability is higher than the threshold
def epsilon_greedy_exploration(x, sess, action_tf, obs, step, num_actions, EPS_END, EPS_DECAY):
	EPS_START = 1.0 # Starting Epsilon
	"""
    Args:
        policy_model (callable): mapping of `obs` to q-values
        obs (np.array): current state observation
        step (int): training step count
        num_actions (int): number of actions available to the agent
    """
	eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)
	if random.random() > eps_threshold: # exploit
		action = sess.run(action_tf, {x: obs})
	else: # explore
		action = random.randrange(num_actions)
	return action

def load_seaquest_env():
	return atari_wrappers.wrap_deepmind(atari_wrappers.make_atari("SeaquestNoFrameskip-v4"), frame_stack=True)


def global_step_tensor(name):
	global_step_tensor = tf.get_variable(
	name, 
	trainable=False, 
	shape=[], 
	initializer=tf.zeros_initializer)
	return global_step_tensor

def training_op(comp_grad_op, optimizer, global_step_tensor):
	return optimizer.minimize(comp_grad_op, global_step = global_step_tensor)

def calculate_q_value(target_q_value, num_actions, greedy_action):
	return tf.reduce_sum(target_q_value * tf.one_hot(greedy_action, num_actions), axis=1, keep_dims=True)
