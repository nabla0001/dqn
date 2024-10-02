"""
Implements the deep Q-network (DQN) as published in [1].

The agent learns to play Atari games end-to-end from the pixel input.

[1] Mnih et al. 2015. "Human-level control through Deep Reinforcement Learning". Nature 518.
"""

# base
import  os
import  numpy as np
import  pickle
from    tqdm import tqdm

import  gym
import  tensorflow as tf

# custom
from preprocessing import StateProcessor
from ExperienceMemory import ExperienceMemory
from tf_utils import float2summary

class DQNagent(object):
    """ DQN agent class."""

    def __init__(self,  env_id      = 'Pong-v3',  # environment
                        q_network   = None,       # FunctionApproximator object
                        epsilon     = 0.05,       # e-greedy: prob. of random action
                        n_frames=4,               # number of frames as state
                        size=(28,28),             # rescaled frame size
                        bbox=(0,0,210,160),       # original (0,0,210,160)
                        gamma=0.99                # discount factor
                 ):
        """
        :param env_id:      -- (str)    Open AI environment Id, e.g. 'Pong-v3'
        :param q_network    -- (class)  Q-function network
        :param epsilon:     -- (float)  exploration probability (epsilon-greedy)

        :returns Qlearner object
        """
        # environment
        self.env_id             = env_id
        self.env_info           = {'env_id': env_id, 'gamma': gamma}

        self.eval_env           = gym.make(self.env_id)  # evaluation env

        # agent params
        self.epsilon            = epsilon

        # function approximator
        self.q_network  = q_network

        # state processor
        self.d_state  = [size[0],size[1],n_frames]
        self.preproc  = {
            'bbox':     bbox,
            'resize':   size
        }

        self.state_processor = StateProcessor(
                size=size,
                bbox=bbox,
                method=tf.image.ResizeMethod.BICUBIC
            )

        # experience replay
        self.experience_memory     = None

        # results
        # list of lists where each sublist contains n_episodes returns/episode lengths
        self.results = {
            'av_score':     [],  # average game score
            'loss':         [],  # bellman loss
            'av_q':         [],  # average predicted Q values
        }

        # global variables
        self.START_TRAIN  = False # after replay buffer is filled
        self.N_FRAMES     = 0
        self.N_GAMES      = 0
        self.N_REPORTS    = 0

    # Methods
    def evaluate(self, n_episodes=100, render=False, random=False):
        """ Evaluates control performance of agent.

        :param n_episodes           -- (int)
        :param render               -- (bool) render environment or not
        :param random               -- (bool) if True evaluates using uniform policy

        :return     results         -- (dict)
                        score      -- (n_episodes np.ndarray) accumulated score per episode
                        score_disc -- (n_episodes np.ndarray) discounted accumulated score (using gamma)
                        score_clip -- (n_episodes np.ndarray) clipped undiscounted score
                        n_frames   -- (n_episodes np.ndarray) episode lengths
                        max_q      -- (n_episodes np.ndarray) average maximum predicted Q-value
        """
        assert self.q_network != None, 'No function approximator specified.'
        assert self.q_network.initialized, '\nVariables in Q-network are not initialized.\n\tCall >> Qnetwork.init_variables()\n'

        # init
        results = {
            'score':            [],
            'score_disc':       [],
            'score_disc_clip':  [],
            'score_clip':       [],
            'n_frames':         [],
            'max_q':            []
        }

        for i_episode in tqdm(range(n_episodes),desc='Evaluating agent ({} episodes).'.format(n_episodes)):

            S_      = self.eval_env.reset()

            # preprocess and replicate initially
            S_      = self.state_processor.process(self.q_network.session,S_)
            stack   = np.stack([S_] * 4, axis=2)

            # results
            score             = 0
            score_disc        = 0
            score_disc_clip   = 0
            score_clip        = 0
            n_frames          = 0
            max_q             = 0
            cnt               = 0

            while True:

                cnt += 1

                if render:
                    self.eval_env.render()  # visualize environment

                # act: greedy wrt Q-values
                if random:
                    qval   = np.nan
                    action = self.eval_env.action_space.sample()
                else:
                    qval        = self.q_network.predict(stack)
                    action      = qval.argmax()

                # observe
                S_, reward, done, info  = self.eval_env.step(action)

                # preprocess and update stack
                S_          = self.state_processor.process(self.q_network.session, S_)
                stack       = np.append(stack[:,:,1:], np.expand_dims(S_, 2), axis=2)

                # record stats
                score               += reward
                score_disc          += self.env_info['gamma']**(cnt-1)*reward
                score_disc_clip     += self.env_info['gamma']**(cnt-1)*np.clip(reward,-1,1)
                score_clip          += np.clip(reward,-1,1)
                max_q               += np.max(qval)
                n_frames            += 1

                # record
                if done:

                    # store results
                    results['score'].append(score)
                    results['score_disc'].append(score_disc)
                    results['score_disc_clip'].append(score_disc_clip)
                    results['score_clip'].append(score_clip)
                    results['max_q'].append(np.mean(max_q))
                    results['n_frames'].append(n_frames)
                    break
                else:
                    pass

        return results

    def train(self,
              sess=None,
              n_episodes=100,     # training episodes
              memory_size=1000,   # replay memory size
              batch_size=32,      # replay batch size
              n_eval_episodes=20, # evaluation
              anneal_epsilon=None, # anneal exploration rate
              n_verbose=1,
              log_dir = '.'):
            """ Learns Q-function online using
                (1) experience replay
                (2) stationary Q-targets

            :param sess           -- Tensorflow session
            :param n_episodes     -- (int)
            :param memory_size    -- (int) replay memory buffer size
            :param batch_size     -- (int) replay batch size
            :param n_eval_episodes-- (int) # episodes for evaluation
            :param anneal_epsilon -- (tuple) epsilon annealing schedule
                                            if None:    fixed
                                            (max,min,steps):  linear annealing from --max to --min over
                                                            --steps (then fixed)
            :param n_verbose      -- (int) evaluation frequency (in frames)
            :param log_dir        -- (str) checkpoint folder

            :return results       -- (dict) containing 'scores', 'durations', 'loss'
            """
            assert self.q_network != None,    'No function approximator to optimize.'

            # make environment
            env             = gym.make(self.env_id)

            self.d_act          = env.action_space.n
            self.org_d_state    = env.observation_space.shape

            # create experience memory
            self.experience_memory = ExperienceMemory(
                            size=memory_size,
                            d_state=self.d_state)

            # epsilon annealing
            if anneal_epsilon != None:
                self.epsilon_annealing_factor = -(anneal_epsilon[0]-anneal_epsilon[1])/float(anneal_epsilon[2])
                self.epsilon_annealing_offset = anneal_epsilon[0]
            else:
                self.epsilon_annealing_factor = 1
                self.epsilon_annealing_offset = 0
                anneal_epsilon                = (self.epsilon,self.epsilon,np.inf)
            # verbose
            print("\n\nTraining DQN agent on Atari's {} ...\n\nEpsilon:\t\t{}\nn_episodes:\t\t{}\nn_eval_episodes:\t{}\nBatch size (replay):\t{}\nMemory size:\t\t{}\n\nEpsilon annealing:\t{} - {} over first {} steps --factor [{}]\nCrop box:\t\t{}\nRescale:\t\t{}\nlog_dir:\t\t{}\n\n".format(
                self.env_id,
                self.epsilon,
                n_episodes,
                n_eval_episodes,
                batch_size,
                memory_size,
                anneal_epsilon[0],anneal_epsilon[1],anneal_epsilon[2],self.epsilon_annealing_factor,
                self.preproc['bbox'],
                self.preproc['resize'],
                log_dir))

            # run session
            sess = self.q_network.session

            for episodeI in range(n_episodes):

                self.N_GAMES += 1

                # for online: environment for learning (avoids interference with evaluation)
                S_              = env.reset()
                S_              = self.state_processor.process(sess, S_)
                stack_S_        = np.stack([S_] * 4, axis=2) # s_(t+1)

                while True: # until termination

                    self.N_FRAMES += 1

                    # act & observe
                    action, S_, reward, done = self.take_egreedy_action(env, stack_S_)

                    # clip rewards
                    reward      = np.clip(reward,-1,1)
                    assert reward >= -1 and reward <= 1, 'Reward must be in[-1,1]. Is {}'.format(reward)

                    # pre-process and update stack
                    stack_S     = stack_S_
                    S_          = self.state_processor.process(sess, S_)
                    stack_S_    = np.append(stack_S_[:,:,1:], np.expand_dims(S_, 2), axis=2) # stack

                    # update memory with current experience
                    self.experience_memory.store(stack_S, action, reward, stack_S_, done)

                    # sample mini-batch from buffer
                    s, a, r, s_, terminal = self.experience_memory.sample(batch_size)

                    # update parameters: gradient descent
                    loss, delta = self.q_network.update(s, a, r, s_, terminal)

                    if done:
                        break

                    if self.N_FRAMES%n_verbose == (n_verbose-1):

                        print('-- {0: 010d} games played [{1: 010d} frames] | saving checkpoint.'.format(self.N_GAMES+1,self.N_FRAMES+1))

                        # evaluate
                        #results = self.evaluate(n_episodes=n_eval_episodes)

                        # save
                        self.q_network.save_ckpt()

                        self.N_REPORTS += 1  # update

            return

    # actions
    def take_egreedy_action(self, env, S):
        """ Take epsilon-greedy step given a state.
        With probability (epsilon) the agent selects a random action, with
        (1 - epsilon) the action yielding the maximum Q-value.

        Additionaly epsilon is annealed linearly during the first n steps.

        :param env      -- (class)      environment
        :param S        -- (np.array)   state stack (H, W, 4)

        :return action  -- (int)        action
        :return S       -- (np.array)   next frame s_(t+1)
        :return reward  -- (int)        reward
        :return done    -- (bool)       termination flag
        """

        # current epsilon
        cur_epsilon =  self.N_FRAMES*(self.epsilon*self.epsilon_annealing_factor) + self.epsilon_annealing_offset

        # log
        #print 'Step {} -- epsilon {}'.format(self.N_FRAMES,cur_epsilon)

        # random (0) or argmax Q-value (1)
        is_greedy = np.random.rand() > cur_epsilon

        # select action
        if is_greedy: # greedy wrt to q-values
            # evaluate current state
            qval = self.q_network.predict(S)

            # greedy: argmax over Q-values
            action = qval.argmax()

        else: # uniform random
            action = env.action_space.sample()

        # take action
        S, reward, done, _ = env.step(action)

        return action, S, reward, done


