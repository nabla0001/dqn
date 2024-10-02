"""
Evaluate all checkpoint(s) of trained Q-learner and
creates a learning curve (offline).

We load the function approximator and compute the game score
on a greedy policy wrt the learning Q-value function.
"""
import  os
import  argparse
import  gym
from    gym import wrappers
import  tensorflow as tf
import  numpy as np


from tf_utils       import loadModel, load_ckpt, plot_std
from DQNagent       import DQNagent
from QNetwork       import QNetwork
from preprocessing  import StateProcessor
from config         import preproc



if __name__ == '__main__':

    # ----- parse command line -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--env','-e', type=str, default='Pong-v3',
                      help='Environment id (default: Pong-v3).')
    parser.add_argument('--n_episodes','-n', type=int, default=1,
                      help='Number of episodes to evaluate control performance.')
    FLAGS, _ = parser.parse_known_args()
    # ------------------------------

    MODEL_PATH      = os.path.join('..','models') # models for exercise B
    f               = os.path.join(MODEL_PATH,FLAGS.env)

    # other params
    GAMMA           = 0.99
    D_STATE         = preproc[FLAGS.env]['D']
    ops_toload      = ['func_approx','state']
    RENDER          = False

    # verbose
    print('\n\nEvaluating Q-learner from {}\n...on {} [across {} episodes, gamma={}]\n\n'.format(
    f, FLAGS.env, FLAGS.n_episodes, GAMMA))

    # init environment
    bbox = preproc[FLAGS.env]['bbox']

    env = gym.make(FLAGS.env)

    # init DQN
    q_network   = QNetwork(env.action_space.n,D_STATE,is_trained=True)
    DQN         = DQNagent(
                    env_id=FLAGS.env,
                    q_network=q_network,
                    size=(D_STATE[0],D_STATE[0]),
                    bbox=bbox,
                    gamma=GAMMA
                )

    DQN.eval_env = env
    DQN.experience_memory = [] # save memory

    # init
    RESULTS = {
        'score':            [],
        'score_disc':       [],
        'score_disc_clip':  [],
        'score_clip':       [],
        'n_frames':         [],
        'max_q':            []
    }

    # restore
    with tf.Session() as sess:

            # load Q-network
            #ops = load_ckpt(sess, ops_toload, f,verbose=False)
            ops = loadModel(sess, ops_toload, f,verbose=False)

            # TF ops
            Q, S =  ops[0],ops[1]

            # assign network to agent
            DQN.q_network.Q           = Q
            DQN.q_network.s           = S
            DQN.q_network.session     = sess

            # evaluate
            results = DQN.evaluate(n_episodes=FLAGS.n_episodes,render=RENDER)

            print('\n-------- Results {} --------'.format(f))
            print('Cumulative score:\t\t\t{} +- {}'.format(np.mean(results['score']),np.std(results['score'])))
            print('Cumulative score (discounted):\t\t{} +- {}'.format(np.mean(results['score_disc']),np.std(results['score_disc'])))
            #print('Cumulative points (clipped):\t\t{} +- {}'.format(np.mean(results['score_clip']),np.std(results['score_clip'])))
            #print('Cumulative points (discounted,clipped):\t{} +- {}'.format(np.mean(results['score_disc_clip']),np.std(results['score_disc_clip'])))
            print('Mean frames:\t\t\t\t{} +- {}\n'.format(np.mean(results['n_frames']),np.std(results['n_frames'])))


