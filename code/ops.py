import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(x,
           n_filters      =16,
           kernel_size    =(3,3),
           stride         =(1,1),
           initializer    =tf.contrib.layers.xavier_initializer(),
           activation_fn  =tf.nn.relu,
           padding        ='VALID',
           name           ='conv2d'):
  """ Convolutional layer

  :param x              -- Tensorflow operator
  :param n_filters      -- (int) number of filters
  :param stride         -- (tuple) convolution stride (x,y)
  :param initializer    -- Tensorflow initializer
  :param activation_fn  -- Tensorflow activation function
  :param padding        -- padding mode
  :param name           -- variable scope
  """
  data_format = 'NHWC' # batch size, heigh, widtht, n_channels

  with tf.variable_scope(name):

    stride        = [1, stride[0], stride[1], 1]
    kernel_shape  = [kernel_size[0], kernel_size[1], x.get_shape()[-1], n_filters]

    # convolution operator
    w             = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
    conv          = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)
    b             = tf.get_variable('biases', [n_filters], initializer=tf.constant_initializer(0.0))
    out           = tf.nn.bias_add(conv, b, data_format)

  if activation_fn != None:
    out = activation_fn(out)

  return out, w, b

def linear(x, out_dim, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
  """ Linear layer

  :param x              -- Tensorflow operator
  :param out_dim        -- (int) # output connections
  :param stddev         -- (float) std for initialization
  :param bias_start     -- (float) bias constant for initialization
  :param activation_fn  -- Tensorflow activation function
  :param name           -- variable scope
  """
  shape = x.get_shape().as_list()

  with tf.variable_scope(name):

    w = tf.get_variable('weights', [shape[1], out_dim], tf.float32,
        tf.random_normal_initializer(stddev=stddev))

    b = tf.get_variable('bias', [out_dim],
        initializer=tf.constant_initializer(bias_start))

    out = tf.nn.bias_add(tf.matmul(x, w), b)

    if activation_fn != None:
      return activation_fn(out), w, b
    else:
      return out, w, b
