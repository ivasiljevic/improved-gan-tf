import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

# Linear
def linear(input_, output_size, scope=None, stddev=0.05, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        
        return tf.matmul(input_, matrix) + bias
        
# Conv2D Layer
def conv2d(input_, out_channels, filter_h=3, filter_w=3, stride_vert=2, stride_horiz=2, stddev=0.05, name="conv2d"):
    with tf.variable_scope(name):
        # Get the number of input channels
        in_channels = input_.get_shape()[-1]
        
        # Construct filter
        w = tf.get_variable('w', [filter_h, filter_w, in_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_vert, stride_horiz, 1], padding='SAME')

        # Add bias
        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        return conv
        
# Deconv2D Layer
def deconv2d(value, output_shape, filter_h=5, filter_w=5, stride_vert=2, stride_horiz=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        # Get the number of input/output channels
        in_channels = value.get_shape()[-1]
        out_channels = output_shape[-1]
        
        # Construct filter
        w = tf.get_variable('w', [filter_h, filter_w, out_channels, in_channels],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(value, w, output_shape=output_shape,
                                        strides=[1, stride_vert, stride_horiz, 1])

        # Add bias
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv
        
# Leaky RELU
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def save_plot(pred,k):
     examples = pred[0:30]#sess.run(pred, feed_dict={x: test, phase_train.name:False})#/255.#.astype(int)
     count = 0
     rngsize = 3
     F = np.zeros((32*rngsize,32*rngsize,3))
     for i in range(rngsize):
         for j in range(rngsize):
             F[i*32:(i+1)*32, j*32:(j+1)*32,:] = np.reshape(examples[count], (32,32,3))
             count = count + 1
     plt.imshow(F, interpolation='nearest')
     plt.savefig("GAN_"+str(k)+".png")
     plt.close()
