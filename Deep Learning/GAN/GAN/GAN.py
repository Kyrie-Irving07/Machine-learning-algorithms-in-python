import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

class G():
    def _init__(self, input_dim, dim_seq):
        self.dim_seq = dim_seq
        self.depth = len(self.dim_seq)
        self.input_dim = self.dim_seq[0]
        self.input = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim])
        self.output = self.fully_connected("Generator", self.input)

    def fully_connected(self, name, input):
        with tf.variable_scope(name) as scope:
            for i in range(self.depth - 1):
                w = tf.get_variable("weight" + str(i), shape = [self.dim_seq[i], self.dim_seq[i + 1]],
                                    dtype = tf.float32, initializer = tf.random_normal_initializer(stddev = 0.1))
                b = tf.get_variable("bias" + str(i), shape = [self.dim_seq[i + 1]])
                input = tf.add(tf.matmul(input, w), b)
                if i == self.depth - 2:
                    input = tf.nn.tanh(input)
                else:
                    input = tf.nn.relu(input)
            return input