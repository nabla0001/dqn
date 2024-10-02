# Tensorflow utility functions

import tensorflow as tf
import numpy      as np

import os
import tensorflow as tf

def loadModel(sess, op_names, checkpoint_path,verbose=True):
	"""Loads TF model.

	Args:
		op_names 		-- list of op names as strings
		checkpoint_path -- path to model folder
	Output:
		saved_ops 	-- list of saved ops (in order of op names)
	"""

	# use latest checkpoint
	ckpt = tf.train.get_checkpoint_state(checkpoint_path)

	if ckpt and ckpt.model_checkpoint_path:
		saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
		saver.restore(sess, ckpt.model_checkpoint_path)

	saved_ops = []
	for i,op in enumerate(op_names):
		saved_ops.append(tf.get_collection(op)[0])

	if verbose:
		print('[LOADED]')
	return saved_ops

def load_ckpt(sess, op_names, checkpoint_file,verbose=True):
	"""Loads TF model checkpint.

	Args:
		op_names 		-- list of op names as strings
		checkpoint_path -- path to model folder
	Output:
		saved_ops 	-- list of saved ops (in order of op names)
	"""

	saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
	saver.restore(sess, checkpoint_file)

	saved_ops = []
	for i,op in enumerate(op_names):
		saved_ops.append(tf.get_collection(op)[0])

	if verbose:
		print('[LOADED]')
	return saved_ops

def saveModel(sess, saver, model_file, ops, step, verbose=True):
	"""Using saver.V1 saves model (works in TF 0.10 and 0.12)
	by saving the important ops as collection.
	Objects can be loaded using function load_TF_model.

	Args:
	    sess 		-- session
	    saver 		-- tf.saver
	    model_file	-- file name (str)
	    ops			-- list of tuples with [(op,'name'), ... ]
	    step		-- (int) global step
        global_step -- optimization steps (tf.variable)
	"""
	assert type(ops) == list, 'ops must be list, is {}'.format(type(ops))

	if verbose:
		print('Saving model {}\t'.format(model_file))

	for t in ops:

		tf.add_to_collection(t[1],t[0])

	saver.save(sess, model_file,global_step=step)

	if verbose:
		print('[SAVED]\n')

	return True

def float2summary(value,tag):
	"""Converts float to summary object."""
	return tf.Summary(value=[
			tf.Summary.Value(tag=tag, simple_value=value)
		])


def gen_batch_indices(N,batch_size,shuffle=True):
    """Creates shuffled indices for batches.

    :param N            -- number of examples
    :param batch_size
    :param shuffle      -- (bool) shuffle order or not

    :return idx         -- (list) containing list of indices for each batch
    """

    # compute batch_size
    n_batches = int(np.ceil(float(N)/batch_size))

    if shuffle:     # shuffle
        idx = np.random.permutation(N)
    else:           # keep order
        idx = range(N)

    start = 0

    # init
    batch_idx = []

    for ii in range(n_batches):

        if start+batch_size >= N:
            stop = N
        else:
            stop = start + batch_size


        # get batch
        batch_idx.append(list(idx[start:stop]))

        start = stop

    return batch_idx

def plot_std(x=[],ax=[], array=[], axis=0, alpha=0.3, col='b'):
    """ Plots mean +- 1 standard deviation.

    :param x       -- x-axis data
    :param ax      -- plot axis
    :param array   -- data (N x M)
    :param axis    -- axis along which to compute mean/std
    :param alpha   -- transparency for shading
    :param color   -- transparency for shading
    """
    m   = np.mean(array,axis=axis)
    sd  = np.std(array,axis=axis)

    lower, upper = m-sd, m+sd

    ax.plot(x,m,lw=2, color=col)
    ax.fill_between(x,lower,upper, alpha=alpha, color=col)
    return
