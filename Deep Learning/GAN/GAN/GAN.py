import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

class G():
    def _init__(self, dim_seq):
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
                    input = tf.sigmoid(input)
                else:
                    input = tf.nn.relu(input)
            return input

class A():
    def _init__(self, dim_seq):
        self.depth = len(dim_seq)
        self.dim_seq = dim_seq
        self.input = tf.placeholder(dtype=tf.float32, shape=dim_seq[0])
        self.output = self.conv("Adversary", self.input)

    def conv(self, name, input):
        with tf.variable_scope(name) as scope:
            for i in range(self.depth - 1):
                w = self.get_filter(i)
                input = tf.nn.conv2d(input=input, w=w, strides=[1,1,1,1], padding="SAME")
                b = self.get_bias(i, [tf.shape(input)[1], tf.shape(input)[2]])
                input = tf.add(input, b)
                if i == self.depth - 2:
                    input = tf.nn.softmax(input)
                else:
                    input = tf.nn.relu(input)

    def get_filter(self, layer):
            w = tf.get_variable("filter"+str(layer), shape=self.dim_seq(layer), dtype=tf.float32,
                                initializer=tf.random_normal_initializer)
            return w

    def get_bias(self, layer, shape):
        return tf.get_variable("bias"+str(layer), shape=shape, dtype=tf.float32,
                               initializer=tf.random_normal_initializer)