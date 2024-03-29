import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import initiate_basic_model, initiate_autoencoder, initiate_dense_model
import util
np.set_printoptions(threshold=np.nan)
#Argsparse
def main(cli_args):
    parser = argparse.ArgumentParser(description="CSCE 496 HW 2, Classify Cifar data")
    parser.add_argument('--input_dir', type=str, default='/work/cse496dl/shared/homework/02', help = 'Numpy datafile input')
    parser.add_argument('--model_dir',type=str,default='./homework_2/',help='directory where model graph and weights are saved')
    parser.add_argument('--epoch' , type=int, default=100, help = "Epoch : number of iterations for the model")
    parser.add_argument('--batch_size', type=int, default=32, help = "Batch Size")
    parser.add_argument('--model', type=int, help=" '1' for basic model, '2' for best model")
    parser.add_argument('--stopCount', type=int, default = 100, help="Number of times for dropping accuracy before early stopping")
    args_input = parser.parse_args(cli_args)

    if args_input.input_dir:
        input_dir = args_input.input_dir
    else:
        raise ValueError("Provide a valid input data path")

    if args_input.model_dir:
        model_dir = args_input.model_dir
    else:
        raise ValueError("Provide a valid model data path")

    if args_input.epoch:
        epochs = args_input.epoch
    else:
        raise ValueError("Epoch value cannot be null and has to be an integer")

    if args_input.batch_size:
        batch_size = args_input.batch_size
    else:
        raise ValueError("Batch Size value cannot be null and has to be an integer")
    
    if args_input.model:
        model = args_input.model
    else:
        raise ValueError("Model selection must not be empty") 

    if args_input.stopCount:
        stop_counter = args_input.stopCount
    else:
        raise ValueError("StopCount have to be an int") 

    input_dir = '/work/cse496dl/shared/homework/02'
    #Make output model dir
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)

    #Load Data
    x = tf.placeholder(tf.float32, [None,32,32,3], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, 100], name='labels')

    #Specify Model
    if(str(model) == '1'):
        train_images, train_labels, test_images, test_labels, val_images, val_labels = util.load_data("")
        _, outputLayer = initiate_basic_model(x)
        #Run Training with early stopping and save output
        counter = stop_counter
        prev_winner = 0
        curr_winner = 0
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        cross_entropy = util.cross_entropy_op(y , outputLayer)
        global_step_tensor = util.global_step_tensor('global_step_tensor')
        train_op = util.train_op_basic(cross_entropy, global_step_tensor, optimizer)
        conf_matrix = util.confusion_matrix_op(y, outputLayer, 100)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            counter = stop_counter
            for epoch in range (epochs):
                if counter > 0:
                    print("Epoch : " + str(epoch))
                    util.training(batch_size, x , y, train_images,
                                train_labels, session,
                                train_op,conf_matrix, 100)
                    accuracy = util.validation(batch_size, x , y, val_images,
                                            val_labels, session,
                                            cross_entropy,
                                            conf_matrix,100)
                    if epoch == 0:
                        prev_winner = accuracy
                        print("Saving.......")
                        saver.save(session, os.path.join("./homework_2/", "homework_2"))
                    else:
                        curr_winner = accuracy
                        if (curr_winner > prev_winner) and (counter > 0):
                            prev_winner = curr_winner
                            print("Saving.......")
                            saver.save(session, os.path.join("./homework_2/", "homework_2"))
                        else:
                            counter -= 1

                    test_accuracy = util.test(batch_size, x , y, test_images,
                            test_labels, session,
                            cross_entropy, conf_matrix, 100)
                    #Calculate the confidence interval
                    value1 , value2 = util.confidence_interval(test_accuracy, 1.96, test_images.shape[0])
                    print("Confidence Interval : " + str(value1) + " , " + str(value2))
                else:
                    break

    elif(str(model) == '2'):
        sparsity_weight = 5e-3
        #Load the data and reshape it
        train_data = np.load(os.path.join(os.path.join(input_dir , 'imagenet_images.npy')))
        train_images, train_labels, test_images, test_labels, val_images, val_labels = util.load_data("")
        #train_data = np.reshape(train_data, [-1,32,32,1])
        #Add noise to the data
        noise_level = 0.2
        x_noise= x + noise_level * tf.random_normal(tf.shape(x))
        code, outputs = initiate_autoencoder(x_noise, 100)
        #Optimizer for Autoencoder
        sparsity_loss = tf.norm(code, ord=1, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.square(outputs - x)) # Mean Square Error
        total_loss = reconstruction_loss + sparsity_weight * sparsity_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(total_loss)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            util.autoencoder_training(x ,code, epochs, batch_size, train_data, sess, train_op)
            saver.save(sess, os.path.join("./homework_2/", "homework_2"))
        print("Done : " + str(code))
        
        _, outputLayer = initiate_dense_model(code)

        #Run Training with early stopping and save output
        counter = stop_counter
        prev_winner = 0
        curr_winner = 0
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        cross_entropy = util.cross_entropy_op(y , outputLayer)
        global_step_tensor = util.global_step_tensor('global_step_tensor')
        #train_op = util.train_op_encoder(cross_entropy, global_step_tensor, optimizer, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "code_layer"))
        train_op = util.train_op_basic(cross_entropy, global_step_tensor, optimizer)
        conf_matrix = util.confusion_matrix_op(y, outputLayer, 100)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            if os.path.isfile(os.path.join("./homework_2/", "homework_2")):
                saver = tf.train.import_meta_graph("homework_2.meta")
                saver.restore(session,"./homework_2/homework_2")
            code_encode = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "code_layer")
            session.run(tf.variables_initializer(code_encode, name="init_encoded_layer"))
            tf.stop_gradient(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "init_encoded_layer"))
            counter = stop_counter
            for epoch in range (epochs):
                if counter > 0:
                    print("Epoch : " + str(epoch))
                    util.training(batch_size, x , y, train_images,
                                train_labels, session,
                                train_op,conf_matrix, 100)
                    accuracy = util.validation(batch_size, x, y, val_images,
                                            val_labels, session,
                                            cross_entropy,
                                            conf_matrix,100)
                    if epoch == 0:
                        prev_winner = accuracy
                        print("Saving.......")
                        saver.save(session, os.path.join("./homework_2/", "homework_2"))
                    else:
                        curr_winner = accuracy
                        if (curr_winner > prev_winner) and (counter > 0):
                            prev_winner = curr_winner
                            print("Saving.......")
                            saver.save(session, os.path.join("./homework_2/", "homework_2"))
                        else:
                            print("Validation Loss : " + str(curr_winner - prev_winner))
                            counter -= 1

                    test_accuracy = util.test(batch_size, x , y, test_images,
                            test_labels, session,
                            cross_entropy, conf_matrix, 100)
                    #Calculate the confidence interval
                    value1 , value2 = util.confidence_interval(test_accuracy, 1.96, test_images.shape[0])
                    print("Confidence Interval : " + str(value1) + " , " + str(value2))
                else:
                    break
    
                
if __name__ == "__main__":
    main(sys.argv[1:])