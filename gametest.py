#!/usr/bin/env python

'''
Author: Sung Jin Hwang
This program uses tensorflow to build a network for playing Flappy Bird.
The wrapped_flappy_bird.py is work of Yen Chen Lin and I used it for training purpose.
This code is also inspired by his work.
Source: https://github.com/yenchenlin/DeepLearningFlappyBird
'''
import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import pandas as pd

GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000 # timesteps to observe before training
EXPLORE = 2000000 # frames over which to reduce random trigger
FINAL_RAND_TRIGGER = 0.0001 # final value of random trigger
INITIAL_RAND_TRIGGER = 0.0001 # starting value of random trigger Change this to 0.1 when training
MEMORY_SIZE = 50000 # number of previous transitions to remember
BATCH = 32 # size of batch

def conv2d(x, weights, stride):
    return tf.nn.conv2d(x, weights, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    '''
    Creates the CNN network. Design is the classical CNN from DeepMinds
    '''
    # network weights
    weights_conv1 = tf.Variable(tf.truncated_normal(shape = [8, 8, 4, 32], stddev = 0.01))
    bias_conv1 = tf.Variable(tf.constant(0.01, shape = [32]))

    weights_conv2 = tf.Variable(tf.truncated_normal(shape = [4, 4, 32, 64], stddev = 0.01))
    bias_conv2 = tf.Variable(tf.constant(0.01, shape = [64]))

    weights_conv3 = tf.Variable(tf.truncated_normal(shape = [3, 3, 64, 64], stddev = 0.01))
    bias_conv3 = tf.Variable(tf.constant(0.01, shape = [64]))

    weights_fc1 = tf.Variable(tf.truncated_normal(shape = [1600, 512], stddev = 0.01))
    bias_fc1 = tf.Variable(tf.constant(0.01, shape = [512]))

    weights_fc2 = tf.Variable(tf.truncated_normal(shape = [512, 2], stddev = 0.01))
    bias_fc2 = tf.Variable(tf.constant(0.01, shape = [2]))

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, weights_conv1, 4) + bias_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights_conv2, 2) + bias_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, weights_conv3, 1) + bias_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_full_connect1 = tf.nn.relu(tf.matmul(h_conv3_flat, weights_fc1) + bias_fc1)

    # readout layer
    Q = tf.matmul(h_full_connect1, weights_fc2) + bias_fc2

    return s, Q

def train(s, Q, sess):
    a = tf.placeholder("float", [None, 2])
    y = tf.placeholder("float", [None])
    Q_action = tf.reduce_sum(tf.multiply(Q, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - Q_action))
    train_step = tf.train.AdamOptimizer(learning_rate = 1e-6).minimize(cost)

    # open up a game state to interact with the game
    game_state = game.GameState()

    # D is a deque to store past 50,000 timeframes
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    image_t, reward_0, terminal = game_state.frame_step(do_nothing)
    image_t = cv2.cvtColor(cv2.resize(image_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, image_t = cv2.threshold(image_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((image_t, image_t, image_t, image_t), axis=2)

    # saving and loading networks. This function is sourced from https://github.com/yenchenlin/DeepLearningFlappyBird
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    #Start Training
    rand_trigger = INITIAL_RAND_TRIGGER
    t = 0
    time_to_fail = np.array([t])

    while True:
        Q_t = Q.eval(feed_dict={s : [s_t]})[0]
        action_t = np.zeros([2])
        action_index = 0

        # Generate random actions if the trigger is high enough
        if random.random() <= rand_trigger:
            print("Random Action Generated")
            action_index = random.randrange(2)
        else:
            action_index = np.argmax(Q_t)
        action_t[action_index] = 1

        print("Time: ",t, " Q: ", Q_t, " Action chosen: ", action_t) # QMAX
        # Reduce trigger
        rand_trigger = update_rand_trigger(rand_trigger, t)

        # Get next game status
        image_t1, s_t1, reward_t, terminal = get_next_status(action_t, s_t, game_state)

        #Save time at failure when game is lost
        if terminal:
            time_to_fail = np.append(time_to_fail, [t])

        # Store the data in deque D
        D.append((s_t, action_t, reward_t, s_t1, terminal))
        if len(D) > MEMORY_SIZE:
            D.popleft()

        if t > OBSERVE:
            # choose batch to use for training
            batch, s_j_batch, action_batch, reward_batch, s_j1_batch = load_batches(D, BATCH)
            
            y_batch = []
            Q_j1_batch = Q.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(batch)):
                terminal = batch[i][4]
                # if terminal, y = -1, else, y = reward + gamma * Q(a',s')
                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * np.max(Q_j1_batch[i]))

            # Run a gradient step
            train_step.run(feed_dict = {y : y_batch, a : action_batch, s : s_j_batch})

        # update time frame and image to current
        s_t = s_t1
        t += 1

        #Save network and time to fail array as dataframe for analysis
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + 'bird' + '-dqn', global_step = t)
            df = pd.DataFrame(time_to_fail)
            df.to_csv('time_to_fail.csv', header=True)



def update_rand_trigger(rand_trigger, t):
    '''
    INPUT: old random action probability, and current time
    OUTPUT: new, decreased random action probability
    '''
    if rand_trigger > FINAL_RAND_TRIGGER and t > OBSERVE:
        rand_trigger = rand_trigger - (INITIAL_RAND_TRIGGER - FINAL_RAND_TRIGGER)/EXPLORE
    return rand_trigger

def get_next_status(action_t, s_t, game_state):
    image_t1_colored, reward_t, terminal = game_state.frame_step(action_t)
    image_t1 = cv2.cvtColor(cv2.resize(image_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, image_t1 = cv2.threshold(image_t1, 1, 255, cv2.THRESH_BINARY)
    image_t1 = np.reshape(image_t1, (80, 80, 1))
    s_t1 = np.append(image_t1, s_t[:, :, :3], axis=2)
    return image_t1, s_t1, reward_t, terminal

def load_batches(D, BATCH):
    '''
    INPUT: deque() that contains the tuples of data, BATCH size of 32
    OUTPUT: data with size 32 that's sampled from deque()
    '''
    batch = random.sample(D, BATCH)
    s_j_batch, action_batch, reward_batch, s_j1_batch = ([] for i in range(4))
    for d in batch:
        s_j_batch.append(d[0])
        action_batch.append(d[1])
        reward_batch.append(d[2])
        s_j1_batch.append(d[3])

    return batch, s_j_batch, action_batch, reward_batch, s_j1_batch

def playGame():
    sess = tf.InteractiveSession()
    s, Q = createNetwork()
    train(s, Q, sess)
    
def main():
    playGame()

if __name__ == "__main__":
    main()