import tensorflow as tf
import collections

from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import RNNCell

def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.99):
    '''Assume 2d [batch, values] tensor'''

    with tf.variable_scope(name_scope):
        size = x.get_shape().as_list()[1]

        scale = tf.get_variable('scale', shape = [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', shape = [size])

        pop_mean = tf.get_variable('pop_mean', shape = [size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', shape = [size], initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)

class BNSRUCell(RNNCell):
    def __init__(self, num_units, training,activation=tf.nn.tanh, state_is_tuple=False, reuse=None):
        super(BNSRUCell, self).__init__(_reuse=reuse)
        self.hidden_dim = num_units
        self.state_is_tuple = state_is_tuple
        self.g = activation
        init_matrix = tf.orthogonal_initializer()

        self.Wr = tf.Variable(init_matrix([self.hidden_dim, self.hidden_dim]))
        self.br = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.U = tf.Variable(init_matrix([self.hidden_dim, self.hidden_dim]))
        self.training = training

    @property
    def state_size(self):
        return self.hidden_dim

    @property
    def output_size(self):
        return self.hidden_dim

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            if self.state_is_tuple:
                (c_prev, h_prev) = state
            else:
                c_prev = state
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(inputs, self.Wf) + self.bf
            )

            f = batch_norm(f,'f',self.training)

            # Reset Gate
            r = tf.sigmoid(
                tf.matmul(inputs, self.Wr) + self.br
            )

            r = batch_norm(r,'r',self.training)

            # Final Memory cell
            c = f * c_prev + (1.0 - f) * tf.matmul(inputs, self.U)

            c = batch_norm(c,'c',self.training)
            # Current Hidden state
            current_hidden_state = r * self.g(c) + (1.0 - r) * inputs
            if self.state_is_tuple:
                return current_hidden_state, LSTMStateTuple(c, current_hidden_state)
            else:
                return current_hidden_state, c

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

