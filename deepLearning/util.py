import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def split_data(data, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1
    """
    size = data.shape[0]
    split_idx = int(proportion * size)
    np.random.shuffle(data)
    return data[:split_idx], data[split_idx:]


def load_data(path): 
	
	#loading images
	images = np.load(os.path.join(str(path) , 'fmnist_train_data.npy'))
	labels = np.load(os.path.join(str(path) , 'fmnist_train_labels.npy'))

	images = images.astype('float32') / 255.0
	labels = labels.astype('float32') / 255.0

	#Spliting data for test and train
	train_images, test_images = split_data(images, 0.9)
	train_labels, test_labels = split_data(labels, 0.9)

	#Splitting training data for validation
	train_images, val_images = split_data(train_images, 0.9)
	train_labels, val_labels = split_data(train_labels, 0.9)
	
	return train_images, train_labels, test_images, test_labels, val_images, val_labels


def confusion_matrix_op(pred, actual_data, num_classes):
	conf_mtx = tf.confusion_matrix(
    			tf.argmax(actual_data, axis=1), 
    			tf.argmax(pred, axis=1), 
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

def training(batch_size, x, y, model, train_images, train_labels, session, train_op, cross_entropy_op):
	for i in range(train_images.shape[0] // batch_size):
		batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
		batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
		_ = session.run(train_op, feed_dict = {x: batch_xs, y: batch_ys})

def validation(batch_size, x, y, model, valid_images, valid_labels, session, cross_entropy_op, confusion_matrix_op, num_classes):
	ce_vals = []
	conf_mxs = []
	accuracy = []
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
		for j in range(num_classes):
			total_acc += conf_matrix[j][j]
		accuracy.append(total_acc)
	avg_valid_ce = sum(ce_vals) / len(ce_vals)
	avg_accuracy = sum(accuracy)/ len(accuracy)
	print("VALID CROSS ENTROPY: " + str(avg_valid_ce))
	print("VALID ACCURACY :" + str(avg_accuracy))
	print("VALID CONFUSION MATRIX:")
	#This prints the values across each class
	print(str(sum(conf_mxs)))
	return avg_accuracy

def test(batch_size, x , y, model, test_images, test_labels, session,cross_entropy_op, confusion_matrix_op):
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
    print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
    print('TEST CONFUSION MATRIX:')
    print(str(sum(conf_mxs)))
