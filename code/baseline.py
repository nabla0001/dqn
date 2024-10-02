"""
Gets baseline scores (random, initialized Q-network) for games.
"""
import matplotlib as mpl
mpl.use('Agg')

import os
import argparse
import gym
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from DQNagent           import DQNagent
from QNetwork           import QNetwork
from config             import preproc

if __name__ == '__main__':

    # ----- parse command line -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--env','-e', type=str, default='Pong-v3',
                      help='Environment id (default: Pong-v3).')
    parser.add_argument('--render','-r', action='store_true', default=False)
    parser.add_argument('--random','-rnd', action='store_true', default=False)
    parser.add_argument('--n_episodes','-n', type=int, default=100)
    FLAGS, _ = parser.parse_known_args()

    # verbose
    print '\n\nEnvironment:\t\t{}'.format(FLAGS.env)

    if FLAGS.random:
        print 'Evaluating random policy ({} episodes)\n\n'.format(FLAGS.n_episodes)
        mode = 'rnd_'
    else:
        print 'Evaluating untrained initialized Q-network ({} episodes)\n\n'.format(FLAGS.n_episodes)
        mode = 'q_init_'
    # ------------------------------

    # environment
    env = gym.make(FLAGS.env)

    # settings for environment
    bbox    = preproc[FLAGS.env]['bbox']
    D_ACT   = env.action_space.n
    D_STATE = [28,28,4]

    # settings
    log_dir = os.path.join('.','logs','baseline_'+mode+FLAGS.env)

    # Q-network
    q_network = QNetwork(
        d_act   = D_ACT,
        d_state = D_STATE,
        f_switch= 5000,
        lr      = 0.00025,
        gamma   = 0.99
    )

    # agent
    DQN = DQNagent(
        env_id = FLAGS.env,
        q_network=q_network,
        epsilon=0.05,
        n_frames=D_STATE[2],           # number of frames as state
        size=(D_STATE[0],D_STATE[0]),  # rescaled frame size
        bbox=bbox,                     # copping bounding box (default does not crop)
    )

    # evaluate
    with tf.Session() as sess:

        # init network
        DQN.q_network.init_variables(sess,log_dir=log_dir)

        # 1. policy based on initialized network
        results = DQN.evaluate(
            n_episodes  =FLAGS.n_episodes,
            render      =FLAGS.render,
            random      =FLAGS.random)

        print '\n-------- Results --------'
        print 'Cumulative score:\t\t\t{} +- {}'.format(np.mean(results['score']),np.std(results['score']))
        print 'Cumulative score (discounted):\t\t{} +- {}'.format(np.mean(results['score_disc']),np.std(results['score_disc']))
        print 'Cumulative points (clipped):\t\t{} +- {}'.format(np.mean(results['score_clip']),np.std(results['score_clip']))
        print 'Cumulative points (discounted,clipped):\t{} +- {}'.format(np.mean(results['score_disc_clip']),np.std(results['score_disc_clip']))
        print 'Mean frames:\t\t\t\t{} +- {}\n'.format(np.mean(results['n_frames']),np.std(results['n_frames']))
