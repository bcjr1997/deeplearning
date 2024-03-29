import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import initiate_basic_model, initiate_better_model
import util

#Argsparse
def main(cli_args):
    parser = argparse.ArgumentParser(description="CSCE 496 HW 1, Classify Fashion MNIST data")
    parser.add_argument('--input_dir', type=str, default='/work/cse496dl/shared/homework/01', help = 'Numpy datafile input')
    parser.add_argument('--model_dir',type=str,default='./homework_1/',help='directory where model graph and weights are saved')
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

    input_dir = '/work/cse496dl/shared/homework/01'
    #Make output model dir
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)

    #Load Data
    train_images, train_labels, test_images, test_labels, val_images, val_labels = util.load_data(input_dir)
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, 10], name='labels')

    #Specify Model
    if(str(model) == '1'):
        _, outputLayer = initiate_basic_model(x)
    elif(str(model) == '2'):
        _, outputLayer = initiate_better_model(x)

    #Run Training with early stopping and save output
    counter = stop_counter
    prev_winner = 0
    curr_winner = 0
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    cross_entropy = util.cross_entropy_op(y , outputLayer)
    global_step_tensor = util.global_step_tensor('global_step_tensor')
    train_op = util.train_op(cross_entropy, global_step_tensor, optimizer)
    conf_matrix = util.confusion_matrix_op(y, outputLayer, 10)
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(10):
            print("KFold : " + str(i))
            counter = stop_counter
            for epoch in range (epochs):
                if counter > 0:
                    print("Epoch : " + str(epoch))
                    util.training(batch_size, x , y, train_images[i],
                                train_labels[i], session,
                                train_op,conf_matrix, 10)
                    accuracy = util.validation(batch_size, x , y, val_images[i],
                                            val_labels[i], session,
                                            cross_entropy,
                                            conf_matrix,10)
                    if epoch == 0:
                        prev_winner = accuracy
                    else:
                        curr_winner = accuracy
                        if (curr_winner > prev_winner) and (counter > 0):
                            prev_winner = curr_winner
                        else:
                            counter -= 1

                    test_accuracy = util.test(batch_size, x , y, test_images[i],
                            test_labels[i], session,
                            cross_entropy, conf_matrix, 10)
                    #Calculate the confidence interval
                    value1 , value2 = util.confidence_interval(test_accuracy, 1.96, test_images[i].shape[0])
                    print("Confidence Interval : " + str(value1) + " , " + str(value2))
                else:
                    break
            print("Saving.......")
            saver.save(session, os.path.join("./homework_1/", "homework_1"))
                
if __name__ == "__main__":
    main(sys.argv[1:])