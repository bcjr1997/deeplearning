import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import initiate_basic_model, initiate_better_model
import atari_wrappers
import gym  #for the RL Environment
import util

#Argsparse
def main(cli_args):
    parser = argparse.ArgumentParser(description="CSCE 496 HW 3, SeaQuest RL Homework")
    parser.add_argument('--n_step', type=int, default=2, help='N-Step time differences for DQN Update Function')
    parser.add_argument('--lambda', type=int, default=0.5, help="Value for Temporal Difference Calculation")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
    parser.add_argument('--model_dir',type=str,default='./homework_3/',help='directory where model graph and weights are saved')
    parser.add_argument('--epoch' , type=int, default=100, help = "Epoch : number of iterations for the model")
    parser.add_argument('--stopCount', type=int, default = 100, help="Number of times for dropping accuracy before early stopping")
    args_input = parser.parse_args(cli_args)

    if args_input.model_dir:
        model_dir = args_input.model_dir
    else:
        raise ValueError("Provide a valid model data path")

    if args_input.epoch:
        epochs = args_input.epoch
    else:
        raise ValueError("Epoch value cannot be null and has to be an integer")

    if args_input.stopCount:
        stop_counter = args_input.stopCount
    else:
        raise ValueError("StopCount have to be an int") 

    #Make output model dir
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)

    #Placeholder for Tensorflow Variables
    x = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, 18], name='output')

    #Load "SeaQuest" from atari_wrapper.py
    seaquest_env = util.load_seaquest_env()
    observation = seaquest_env.reset()
    print(f"Environment Reset : {observation}")
    observation = seaquest_env.step(1)
    print(f"Environment Step : {observation}")
    NUM_ACTIONS = seaquest_env.action_space.n
    OBS_SHAPE = seaquest_env.observation_space.shape

if __name__ == "__main__":
    main(sys.argv[1:])