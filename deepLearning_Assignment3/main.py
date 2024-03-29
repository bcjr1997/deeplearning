import argparse
import os
import parser
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import initiate_better_policy_model, initiate_policy_model, initiate_target_model, initiate_better_target_model
import atari_wrappers
import gym  #for the RL Environment
import util
import random
import itertools

#Argsparse
def main(cli_args):
    parser = argparse.ArgumentParser(description="CSCE 496 HW 3, SeaQuest RL Homework")
    parser.add_argument('--n_step', type=int, default=2, help='N-Step time differences for DQN Update Function')
    parser.add_argument('--lambda', type=int, default=0.5, help="Value for Temporal Difference Calculation")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
    parser.add_argument('--model_dir',type=str,default='./homework_3/',help='directory where model graph and weights are saved')
    parser.add_argument('--epoch' , type=int, default=100, help = "Epoch : number of iterations for the model")
    parser.add_argument('--model', type=int, help=" '1' for basic model, '2' for best model")
    parser.add_argument('--stopCount', type=int, default = 100, help="Number of times for dropping accuracy before early stopping")
    args_input = parser.parse_args(cli_args)

    if args_input.model:
        model = args_input.model
    else:
        raise ValueError("Model selection must not be empty") 

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

    #Make output model dir
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)

    #Placeholder for Tensorflow Variables
    x = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_placeholder') #4 frames
    y = tf.placeholder(tf.float32, [None, 18], name='output') #18 possible outputs
    
    #Setup
    LEARNING_RATE = 0.0001
    TARGET_UPDATE_STEP_FREQ = 5
    number_of_episodes = epochs

    replay_memory = util.ReplayMemory(10000)
    #Optimizer
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
    #Load "SeaQuest" from atari_wrapper.py
    seaquest_env = util.load_seaquest_env()
    NUM_ACTIONS = seaquest_env.action_space.n #18 Possible Actions
    OBS_SHAPE = seaquest_env.observation_space.shape # [height, width, channels] = [84, 84, 4]
    EPS_END = 0.1
    EPS_DECAY = 100000
    step = 0

    if(str(model) == '1'):
        policy_model, policy_output_layer = initiate_policy_model(x, NUM_ACTIONS)
        target_model, target_output_layer = initiate_target_model(x, NUM_ACTIONS)
        print("Basic Model Initialized")
    elif(str(model) == '2'):
        policy_model, policy_output_layer = initiate_better_policy_model(x, NUM_ACTIONS)
        target_model, target_output_layer = initiate_better_target_model(x, NUM_ACTIONS)
        print("Better Model Initialized")
    prev_episode_score = -1
    saver = tf.train.Saver()
    argmax_action = tf.argmax(policy_output_layer, axis = 1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for episode in range(number_of_episodes):
            prev_observation = seaquest_env.reset()
            observation, reward, done, _ = seaquest_env.step(random.randrange(NUM_ACTIONS))
            done = False
            episode_score = 0.0
            while not done:
                prep_obs = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
                curr_action = util.epsilon_greedy_exploration(x, sess, argmax_action, prep_obs, step, NUM_ACTIONS, EPS_END, EPS_DECAY)
                observation, reward, done, _ = seaquest_env.step(curr_action)
                next_obs = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
                next_action = util.epsilon_greedy_exploration(x, sess, argmax_action, next_obs, step, NUM_ACTIONS, EPS_END, EPS_DECAY)
                following_observation, next_reward, next_done , _ = seaquest_env.step(next_action)
                replay_memory.push(prev_observation, curr_action, observation, reward, next_action, next_reward, following_observation) # s, a , r, s' a' r' s'' 
                prev_observation = observation

                #run the session with the placeholders
                loss, gradients= util.dqn_gradient_calculation(replay_memory, x, y, policy_output_layer, target_output_layer, sess, batch_size, optimizer)
                episode_score += next_reward
                step += 1
            print(f"Episode : {episode} Episode Score : {episode_score} Step: {step}")
            if episode % TARGET_UPDATE_STEP_FREQ == 0:
                for (target_var, policy_var) in zip(tf.trainable_variables(target_model), tf.trainable_variables(policy_model)):
                    tf.assign(target_var, policy_var)
                if episode_score >= prev_episode_score:
                    print("Saving .........")
                    saver.save(sess, os.path.join("./homework_3/", "homework_3"))
                    prev_episode_score = episode_score
            if episode_score >= prev_episode_score:
                print("Saving .........")
                saver.save(sess, os.path.join("./homework_3/", "homework_3"))
                prev_episode_score = episode_score

if __name__ == "__main__":
    main(sys.argv[1:])