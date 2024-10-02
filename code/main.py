"""
Runs DQN experiments
"""

import os
import argparse
from   datetime import datetime
import gym
import tensorflow as tf

from DQNagent           import DQNagent
from QNetwork           import QNetwork
from config             import preproc

if __name__ == '__main__':

    # ----- parse command line -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--env','-e', type=str, default='Pong-v3',
                      help='Environment id (default: Pong-v3).')
    FLAGS, _ = parser.parse_known_args()

    time_str = datetime.now().strftime('%H%M%S')
    # ------------------------------

    # environment
    env = gym.make(FLAGS.env)

    # settings for environment
    bbox    = preproc[FLAGS.env]['bbox']
    D_ACT   = env.action_space.n
    D_STATE = [84,84,4]


    # settings
    log_dir = os.path.join('..','logs',FLAGS.env+'_'+time_str)

    # Q-network
    q_network = QNetwork(
        d_act   = D_ACT,
        d_state = D_STATE,
        f_switch= 10000,     # target network switch frequency
        lr      = 0.00025,   # learning rate
        gamma   = 0.99       # discount factor
    )

    # agent
    DQN = DQNagent(
        env_id = FLAGS.env,
        q_network=q_network,
        epsilon=0.1,                        # exploration rate
        n_frames=D_STATE[2],                # number of frames as state
        size=(D_STATE[0],D_STATE[0]),       # rescaled frame size
        bbox=bbox,                          # copping bounding box (default does not crop)
    )

    with tf.Session() as sess:

        # init network
        DQN.q_network.init_variables(sess,log_dir=log_dir)

        # train
        DQN.train(
                  n_episodes=50000,     # training episodes
                  memory_size=100000,   # replay memory size
                  batch_size=32,        # replay batch size
                  n_eval_episodes=20,   # evaluation
                  n_verbose=50000,
                  anneal_epsilon=(1,0.1,1000000),
                  log_dir = log_dir)


