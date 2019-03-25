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
import random

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

    if args_input.batch_size:
        batch_size = args_input.batch_size

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
    x = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_placeholder') #4 frames
    y = tf.placeholder(tf.float32, [None, 18], name='output') #18 possible outputs

    #Setup
    LEARNING_RATE = 0.0001
    TARGET_UPDATE_STEP_FREQ = 10
    number_of_episodes = 20
    policy_model = initiate_basic_model(x)
    target_model = initiate_basic_model(x)
    replay_memory = util.ReplayMemory(1000000)
    #Optimizer
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
    #Load "SeaQuest" from atari_wrapper.py
    seaquest_env = util.load_seaquest_env()
    NUM_ACTIONS = seaquest_env.action_space.n
    OBS_SHAPE = seaquest_env.observation_space.shape
    EPS_END = 0.1
    EPS_DECAY = 100000
    step = 0
    for episode in range(number_of_episodes):
        prev_observation = seaquest_env.reset()
        observation, reward, status, info = seaquest_env.step(random.randrange(NUM_ACTIONS))
        done = False
        episode_score = 0

        while not done:
            prep_obs = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
            curr_action = util.epsilon_greedy_exploration(policy_model, prep_obs, step, NUM_ACTIONS, EPS_END, EPS_DECAY)
            observation, reward, done, _ = seaquest_env.step(curr_action)

            replay_memory.push(prev_observation, curr_action, observation, reward)
            prev_observation = observation

            loss, gradients = util.dqn_gradient_calculation(replay_memory, policy_model, target_model, batch_size, optimizer)
            if gradients is not None:
                optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

            episode_score += reward
            step += 1

        if episode % TARGET_UPDATE_STEP_FREQ == 0:
            target_model

if __name__ == "__main__":
    main(sys.argv[1:])