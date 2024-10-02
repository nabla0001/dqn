"""
Preprocessing function for DQN agent. For details see DQN paper [1]

[1] Mnih et al. 2015. "Human-level control through Deep Reinforcement Learning". Nature 518.
"""

import numpy as np
import tensorflow as tf

class StateProcessor():
    """
    Processes a raw Atari frame by (1) resizing and (2) converting to grayscale
    """
    def __init__(self, size=(28,28), bbox=None, method=tf.image.ResizeMethod.BICUBIC):
        """

        :param n_frames:-- (int) number of frames stacked as state
        :param size:    -- (tuple) size of rescaled images
        :param bbox:    -- (tuple) crops to bbox (vertical ul, horizontal, ul, height, width)
        :param method:  -- tensorflow resizing method (BICUBIC, NEAREST_NEIGHBOR)

        Note: this only works for single frames.
        """
        # Build the Tensorflow graph
        with tf.variable_scope("State_processor"):

            self.input_state    = tf.placeholder(shape=[210,160,3], dtype=tf.float32)

            self.output         = self.input_state / 255.0

            if bbox != None:
                self.output     = tf.image.crop_to_bounding_box(self.output, bbox[0], bbox[1], bbox[2], bbox[3])

            self.output         = tf.image.rgb_to_grayscale(self.output)

            self.output         = tf.image.resize_images(
                                        self.output, size, method=method)

            self.output         = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        :param sess:       --  tensorflow session object
        :param state:      -- (np.array)  [210, 160, 3] Atari RGB State

        :return: processed [N, M, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


if __name__ == '__main__':

    import argparse
    import gym
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from config import preproc

    # ----- parse command line -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--env','-e', type=str, default='Pong-v3',
                      help='Environment id (default: Pong-v3).')
    parser.add_argument('--plot','-p', action='store_true', default=False)

    FLAGS, _ = parser.parse_known_args()
    # ------------------------------
    # test folder
    TEST_DIR = './tests_' + FLAGS.env
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)

    # test preprocessing
    env = gym.make(FLAGS.env)

    s_  = env.reset()

    # processor
    size        = (48,48)
    bbox        = preproc[FLAGS.env]['bbox']
    processor   = StateProcessor(size=size, bbox=bbox)
    #processor   = StateProcessor(size=size, bbox=bbox,method=tf.image.ResizeMethod.BILINEAR)

    with tf.Session() as sess:

        for t in range(75):

            if FLAGS.plot:
                env.render()

            action = env.action_space.sample()

            s                 = s_
            s_, r, done, info = env.step(action)

            # preprocess
            #s_processed = preprocess(s_, threshold=0.4)
            s_processed  = processor.process(sess, s_)

            # save results
            if FLAGS.plot :
                file_name    = os.path.join(TEST_DIR,'frame{0: 04d}.pdf'.format(t))
                f, ((ax1,ax2)) = plt.subplots(1,2)

                # original
                ax1.imshow(s_)
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_title('Original')

                # processed
                ax2.imshow(s_processed)
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.set_title('Processed')

                plt.tight_layout()
                plt.savefig(file_name)
                plt.close()

            if done:
                break


