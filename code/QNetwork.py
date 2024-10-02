"""
Implements Q-function approximator with a target network in tensorflow

The graph is only compatible with Tensorflow version r0.10
"""

import os
import numpy as np
from functools import reduce
import tensorflow as tf

from tf_utils   import saveModel
from ops        import conv2d, linear, clipped_error

class QNetwork(object):
    """Q-network: action-value function approximator.

    Implements the compute graph prediction action-values.
    Consists of a periodically updated Q-target network and Q-network.

    The parameters of the networks are stored in:
        self.w        # Q network
        self.t_w      # Q_ network (target network)

    :param d_state  -- (list)  state space dimensionality, e.g. [210, 160, 3] for atari
    :param d_act    -- (list)  action space dimensionality
    :param f_switch -- (int)  frequency of updating target network updates
    :param lr       -- (float)  starting learning rate
    :param gamma    -- (float)  discount factor of MDP
    :param is_trained -- (bool)  set to True for evaluation after training
    """

    def __init__(self,d_act, d_state=[28,28,4], f_switch=5, lr=0.00025, huber_loss = True, gamma=0.99, is_trained=False):

        assert type(d_state) == list and type(d_act) == int, 'Type error: d_state {} (must be list), d_act {} (must be int)'.format(d_state,d_act)
        assert len(d_state) == 3, 'd_state should have 3 entires (screen_height, screen_width, n_frames)'

        d_state, D_ACT = d_state, d_act # dimensionality state space, action space

        # session
        self.session        = None
        self.N_STEP         = 0 # counts iterations
        self.f_switch       = f_switch
        self.d_act          = d_act
        self.d_state        = d_state
        self.initialized    = False

        # training hyper params
        self.scale                       = 30000
        self.learning_rate               = lr #0.00025
        self.learning_rate_minimum       = 0.00001
        self.learning_rate_decay         = 0.96
        self.learning_rate_decay_step    = 5 * self.scale

        self.initializer    = tf.truncated_normal_initializer(0, 0.02)
        self.activation_fn  = tf.nn.relu

        # parameters Q-network
        self.w      = {}
        self.t_w    = {} # target network

        #if huber_loss:
            #print('\n[Q-network:] Using huber loss for optimization.\n')

        if not is_trained:
            # placeholders
            with tf.name_scope('Input'):
                self.s        = tf.placeholder(tf.float32,    [None, d_state[0], d_state[1],  d_state[2]],     name='state')
                self.s_       = tf.placeholder(tf.float32,    [None, d_state[0], d_state[1],  d_state[2]],     name='state_')
                self.r        = tf.placeholder(tf.float32,    [None],                     name='reward')
                self.a        = tf.placeholder(tf.int32,      [None],                     name='action')
                self.terminal = tf.placeholder(tf.float32,    [None],                     name='terminal')

            # function approximator for q-function
            with tf.variable_scope('Function_approximator'):
                self.build_network()

                 # Q-targets: R_(t+1) + gamma * max_a Q(s_(t+1),a)
                with tf.name_scope('Q_Targets'):
                    self.q_target   = tf.reduce_max(self.Q_,reduction_indices=1)
                    self.q_targets  = tf.add(self.r, tf.multiply(gamma * self.q_target, 1-self.terminal))  # uses only reward on terminal state

                with tf.name_scope('Q_Predictions'):
                    self.a_one_hot      = tf.one_hot(self.a, self.d_act, 1.0, 0.0, name='action_one_hot')
                    self.q_predictions  = tf.reduce_sum(self.Q * self.a_one_hot, reduction_indices=1, name='q_acted')

            # operator for copying params to target network
            with tf.variable_scope('Pred_to_target'):
                self.t_w_input = {}
                self.t_w_assign_op = {}

                for name in self.w.keys():
                    self.t_w_input[name]        = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                    self.t_w_assign_op[name]    = self.t_w[name].assign(self.t_w_input[name]) # copy params -> target_params

            # Optimization
            with tf.name_scope('Optimizer'):

                # loss
                with tf.name_scope('Loss'):
                    self.delta      = tf.stop_gradient(self.q_targets) - self.q_predictions

                    if huber_loss:
                        self.loss       = tf.reduce_mean(clipped_error(self.delta), name = 'loss')
                    else:
                        self.loss       = tf.reduce_mean(tf.square(self.delta), axis=0,name='loss')

                # learning rate schedule
                with tf.name_scope('Learning_rate_schedule'):
                    self.global_step = tf.Variable(0, trainable=False)

                    self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
                    self.learning_rate_op   = tf.maximum(self.learning_rate_minimum,
                      tf.train.exponential_decay(
                          self.learning_rate,
                          self.learning_rate_step,
                          self.learning_rate_decay_step,
                          self.learning_rate_decay,
                          staircase=True))
                    self.optimizer = tf.train.RMSPropOptimizer(
                     self.learning_rate_op, momentum=0.95, epsilon=0.01)
                    self.train_op = self.optimizer.minimize(self.loss,global_step=self.global_step)
        else:
            self.initialized = True


    def build_network(self):
        """ Builds Q-network (prediction and target network)

        All parameters are stored in
            self.w
            self.t_w
        """
        with tf.variable_scope('Prediction_network'):
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s,
              32, (8, 8), (4, 4), self.initializer, self.activation_fn, name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
              64, (4, 4), (2, 2), self.initializer, self.activation_fn, name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
              64, (3, 3), (1, 1), self.initializer, self.activation_fn, name='l3')

            # fully connected layers
            shape           = self.l3.get_shape().as_list()

            self.l3_flat                                = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.l4, self.w['fc1_w'], self.w['fc1_b']   = linear(self.l3_flat, 512, activation_fn=self.activation_fn, name='fc1')
            self.Q, self.w['q_w'], self.w['q_b']        = linear(self.l4, self.d_act, name='q_values')

        with tf.variable_scope('Target_network'):
            self.t_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.s_,
              32, (8, 8), (4, 4), self.initializer, self.activation_fn, name='t_l1')
            self.t_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.t_l1,
              64, (4, 4), (2, 2), self.initializer, self.activation_fn, name='t_l2')
            self.t_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.t_l2,
              64, (3, 3), (1, 1), self.initializer, self.activation_fn, name='t_l3')

            # fully connected layers
            shape           = self.t_l3.get_shape().as_list()

            self.t_l3_flat                                  = tf.reshape(self.t_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.t_l4, self.t_w['fc1_w'], self.t_w['fc1_b'] = linear(self.t_l3_flat, 512, activation_fn=self.activation_fn, name='t_fc1')
            self.Q_, self.t_w['q_w'], self.t_w['q_b']       = linear(self.t_l4, self.d_act, name='target_q_values')

        return

    def update_target_q_network(self):
        """ Copies parameters from Q-network to target network."""
        for name in self.w.keys():

            # before
            # print(\n\n------- Key: {} -------'.format(name))
            # print([before] Q-target network:\t', self.t_w[name].eval()[:5])
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

            # check
            # print(Q-network:\t\t', self.w[name].eval()[:5])
            # print([after] Q-target network:\t', self.t_w[name].eval()[:5])
            # print(\n\n')

    def init_variables(self,sess, log_dir='.'):
        """ Initializes graph variables and attaches session.

        :param sess: tensorflow session
        """
        # attach
        self.session        = sess
        self.log_dir        = log_dir

        # create log folder
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # init variables (r0.10)
        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)

        # saver and writer
        self.saver     = tf.train.Saver(max_to_keep=1800)

        # set up summaries
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate_op)
        tf.summary.scalar("max_q", tf.reduce_mean(tf.reduce_max(self.q_predictions)))
        tf.summary.histogram("delta", self.delta)
        tf.summary.histogram("q_targets", self.q_targets)
        tf.summary.histogram("q_predictions", self.q_predictions)
        self.merged_op  = tf.summary.merge_all()

        self.writer    = tf.summary.FileWriter(log_dir,self.session.graph)

        self.initialized = True

        print('\nLogging to {}\n'.format(log_dir))

    def update(self, s, a, r, s_, terminal):
        """ Updates parameters of function approximator.

        Note: FunctionApproximator must have session attached.

        :param s:   state
        :param a:   action (int)
        :param r:   reward
        :param s_:  successor state
        :return: batch loss (float), delta (array)
        """
        assert self.session != None, 'FunctionApproximator must have session attached (is {})'.format(self.session)

        # gradient descent
        _, loss, delta,summaries,lr = self.session.run([self.train_op, self.loss, self.delta, self.merged_op,self.learning_rate_op],
                                            feed_dict={
                                             self.s:    s,
                                            self.s_:    s_,
                                            self.r:     r,
                                            self.a:     a,
                                            self.terminal:     terminal,
                                            self.learning_rate_step: self.N_STEP,
                                                    })

        #print(\nDelta ', delta.shape)
        #print(Q ', q.shape)
        #print(Q_ ', q_.shape)

        if self.N_STEP%self.f_switch == (self.f_switch-1):

            #print('STEP {0}\t[Updating target network.]\tLearning rate:\t{1: .4e}'.format(
            #self.N_STEP+1,lr))

            self.update_target_q_network()

        self.writer.add_summary(summaries,self.N_STEP)

        # update iteration counter
        self.N_STEP += 1

        return loss, delta

    def predict(self, s):
        """ Predicts Q-values given single state.

        :param s  -- (array) single frame stack
        :return Q -- (array) Q-values
        """
        Q = self.session.run(self.Q,  # q-predictions
                feed_dict={self.s: np.expand_dims(s,axis=0)}  # we expand for batch index
                )

        return Q

    def save_ckpt(self):
        """Saves model checkpoint to log_dir."""

        ops_tosave = [(self.Q,'func_approx'),(self.s,'state')]

        saveModel(self.session,
                  self.saver,
                  os.path.join(self.log_dir,'model'),
                  ops_tosave,
                  step=self.global_step,
                  verbose=False)


if __name__ == '__main__':

    net = QNetwork(2,[28,28,4])

    with tf.Session() as sess:

        net.update_target_q_network()

        # create state
        s = np.random.rand(1,28,28,4)
        a = np.array([1])
        r = np.array([-1])
        s_ = s
        t = np.array([1],dtype=float)

        q = net.predict(s)

        print(q)

        # update
        net.update(s,a,r,s_,t)

