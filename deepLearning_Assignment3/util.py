import collections
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import atari_wrappers

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
def huber_loss(x_placeholder, delta=1.0):

	return tf.where(
        tf.abs(x_placeholder) < delta,
        tf.square(x_placeholder) * 0.5,
        delta * (tf.abs(x_placeholder) - 0.5 * delta)
    )

# Calculate the gradients for DQN
def dqn_gradient_calculation(replay_memory, policy_model, target_model, batch_size, optimizer, gamma=0.99, grad_norm_clipping=1.0):
	#Check to see if there are enough transistions to form a batch
	if len(replay_memory) < batch_size:
		return None, None
	#If meet batch size, start training batch
	transistions = replay_memory.sample(batch_size) #Get training data with a batch size
	batch = Transition(*zip(*transistions))
	next_state_batch = np.array(batch.next_state, dtype=np.float32)
	state_batch = np.array(batch.state, dtype=np.float32)
	action_batch = np.array(batch.action, dtype=np.int64)
	reward_batch = np.array(batch.reward)
	following_state_batch = np.array(batch.following_state, dtype=np.float32)
	next_action_batch = np.array(batch.next_action, dtype=np.int64)
	next_reward_batch = np.array(batch.next_reward)

	#Calculate gradient of the graph
	# Calculate values from the action state
	action_index = np.stack([np.arange(batch_size, dtype=np.int32), next_action_batch], axis=1)
	# Get the values of all states 
	print(f"DANK : {policy_model}/")
	state_values = tf.gather_nd(policy_model(next_state_batch), action_index) #True Values
		
	# calculate best value at next state
	next_state_values = tf.reduce_max(target_model(following_state_batch), axis=1)
	# compute the expected Q values
	expected_state_action_values = reward_batch + (gamma * next_reward_batch) + ((gamma*gamma)* next_state_values)

	#Compute Huber Loss with TD error
	td_error = state_values - expected_state_action_values # True values - predicted values
	curr_loss = huber_loss(td_error)
	# Calculate gradient loss
	gradients = optimizer.compute_gradients(curr_loss)

	#Clip gradients
	for index, gradient in enumerate(gradients):
		if gradient is not None:
			gradients[index] = tf.clip_by_norm(gradient, grad_norm_clipping)

	return curr_loss, gradients


# Method that exploits the system if the probability is higher than the threshold
def epsilon_greedy_exploration(policy_model, obs, step, num_actions, EPS_END, EPS_DECAY):
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
		action = tf.argmax(policy_model(tf.convert_to_tensor(obs)), axis=1)
	else: # explore
		action = random.randrange(num_actions)
	return action

def load_seaquest_env():
	return atari_wrappers.wrap_deepmind(atari_wrappers.make_atari("SeaquestNoFrameskip-v4"), frame_stack=True)

def split_data(data, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1
    """
    size = data.shape[0]
    split_idx = int(proportion * size)
    return data[:split_idx], data[split_idx:]

def confidence_interval(accuracy, constant, n):
	error = 1 - accuracy
	calculation = (error * (1 - error)) / n
	value1 = error + (constant * math.sqrt(calculation))
	value2 = error - (constant * math.sqrt(calculation))
	return value1, value2

def one_hot_encoding(labels, num_classes):
	return np.eye(num_classes)[labels.astype(int)]

def load_data_kfold(images,labels,kfold):
	#KFold
	image_partition = []
	label_partition = []
	image_size = int(images.shape[0] / kfold)
	label_size = int(labels.shape[0] / kfold)

	#Split into K Partitions
	for i in range(kfold):
		a = i * image_size
		b = (i + 1) * image_size
		temp_images = images[ a : b ]
		image_partition.append(temp_images)
		a = i * label_size
		b = (i + 1) * label_size
		temp_labels = labels[a : b]
		label_partition.append(temp_labels)

	image_partition = np.array(image_partition)
	label_partition = np.array(label_partition)

	#Shuffle into 10 ways
	train_images = []
	train_labels = []
	test_images = []
	test_labels = []
	val_images = [] 
	val_labels = []
	for i in range(kfold):
		#Get a set for test
		test_images.append(image_partition[i])
		test_labels.append(label_partition[i])
		#Get Validation dataset

		val = i + 1
		if val != 10:
			val_images.append(image_partition[val])
			val_labels.append(label_partition[val])
		else:
			val = 0
			val_images.append(image_partition[val])
			val_labels.append(label_partition[val])

		#Then the remainder will be populated into train
		remain_images = np.array([]).reshape(0,784)
		remain_labels = np.array([]).reshape(0,10)
		for j in range(kfold):
			if i != j and val!= j:
				remain_images = np.concatenate((remain_images, image_partition[j]))
				remain_labels = np.concatenate((remain_labels, label_partition[j]))

		train_images.append(remain_images)
		train_labels.append(remain_labels)

	return train_images, train_labels, test_images, test_labels, val_images, val_labels


def load_data(path): 
	
	#loading images
	images = np.load('fmnist_train_data.npy')
	labels = np.load('fmnist_train_labels.npy')
	permutation = np.random.permutation(images.shape[0])
	images = images[permutation]
	labels = labels[permutation]
	images = images / 255.0
	#print(labels.shape)
	labels = one_hot_encoding(labels,10)
	#print(labels)
	labels = labels.astype(float)
	train_images, train_labels, test_images, test_labels, val_images, val_labels = load_data_kfold(images,labels,10)
	#Get Validation Dataset
	return train_images, train_labels, test_images, test_labels, val_images, val_labels


def confusion_matrix_op(y, output, num_classes):
	conf_mtx = tf.confusion_matrix(
    			tf.argmax(y, axis=1), 
    			tf.argmax(output, axis=1), 
    			num_classes=num_classes)	
	return conf_mtx

def cross_entropy_op(y_placeholder, output):
	return tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=output)

def train_op(cross_entropy_op, global_step_tensor, optimizer):
	return optimizer.minimize(cross_entropy_op, global_step=global_step_tensor)

#Declaring global step tensor
def global_step_tensor(name):
	global_step_tensor = tf.get_variable(
	name, 
	trainable=False, 
	shape=[], 
	initializer=tf.zeros_initializer)
	return global_step_tensor

def training(batch_size, x, y, train_images, train_labels, session, train_op, confusion_matrix_op, num_classes):
	conf_mxs =[]
	avg_accuracy = 0
	for i in range(int(train_images.shape[0]) // batch_size):
		batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
		batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
		
		_,conf_matrix = session.run([train_op, confusion_matrix_op], feed_dict = {x: batch_xs, y: batch_ys})
		conf_mxs.append(conf_matrix)
	avg_conf_mxs= sum(conf_mxs)
	for i in range (num_classes):
		avg_accuracy += avg_conf_mxs[i][i]
	print("TRAIN ACCURACY :" + str(avg_accuracy/train_images.shape[0]))
	print("TRAIN CONFUSION MATRIX:")
	#This prints the values across each class
	print(str(sum(conf_mxs)))

def validation(batch_size, x, y, valid_images, valid_labels, session, cross_entropy_op, confusion_matrix_op, num_classes):
	ce_vals = []
	conf_mxs = []
	for i in range (valid_images.shape[0] // batch_size):
		batch_xs = valid_images[i * batch_size:(i + 1) * batch_size, :]
		batch_ys = valid_labels[i * batch_size:(i + 1) * batch_size, :]
		valid_ce, conf_matrix = session.run(
            [tf.reduce_mean(cross_entropy_op), confusion_matrix_op],
			feed_dict = {
                x: batch_xs,
                y: batch_ys
            })
		ce_vals.append(valid_ce)
		conf_mxs.append(conf_matrix)
	avg_conf_mxs= sum(conf_mxs)
	avg_valid_ce = sum(ce_vals) / len(ce_vals)
	avg_accuracy = 0
	for i in range (num_classes):
		avg_accuracy += avg_conf_mxs[i][i]
	print("VALID CROSS ENTROPY: " + str(avg_valid_ce))
	print("VALID ACCURACY :" + str(avg_accuracy/valid_images.shape[0]))
	print("VALID CONFUSION MATRIX:")
	#This prints the values across each class
	print(str(sum(conf_mxs)))
	return avg_accuracy/valid_images.shape[0]

def test(batch_size, x , y, test_images, test_labels, session, cross_entropy_op, confusion_matrix_op, num_classes):
	# report mean test loss
    ce_vals = []
    conf_mxs = []
    for i in range(test_images.shape[0] // batch_size):
        batch_xs = test_images[i * batch_size:(i + 1) * batch_size, :]
        batch_ys = test_labels[i * batch_size:(i + 1) * batch_size, :]
        test_ce, conf_matrix = session.run(
            [tf.reduce_mean(cross_entropy_op), confusion_matrix_op], {
                x: batch_xs,
                y: batch_ys
            })
        ce_vals.append(test_ce)
        conf_mxs.append(conf_matrix)
    avg_test_ce = sum(ce_vals) / len(ce_vals)
    avg_accuracy = 0
    avg_conf_mxs = sum(conf_mxs)
    for i in range(num_classes):
        avg_accuracy += avg_conf_mxs[i][i]
    print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
    print("TEST ACCURACY :" + str(avg_accuracy/test_images.shape[0]))
    print('TEST CONFUSION MATRIX:')
    print(str(sum(conf_mxs)))
    return avg_accuracy/test_images.shape[0]
