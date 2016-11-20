import tensorflow as tf
import numpy as np
from ops import *

batch_size = 32
sample_size = 32

z_dim = 100
gf_dim = 64
df_dim = 64

learning_rate = 0.0002
beta1 = 0.5

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')
  
def discriminator(image, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
    h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
    #h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
    h3 = lrelu(d_bn3(conv2d(h1, df_dim*4, name='d_h3_conv')))
    h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

    return tf.nn.sigmoid(h4), h4

def generator(z):
    z_ = linear(z, gf_dim*8*4*4, 'g_h0_lin')
    h0 = tf.nn.relu(g_bn0(tf.reshape(z_, [-1, 4, 4, gf_dim * 8])))
    h1 = tf.nn.relu(g_bn1(deconv2d(h0, [batch_size, 8, 8, gf_dim*4], name='g_h1')))
    h2 = tf.nn.relu(g_bn2(deconv2d(h1, [batch_size, 16, 16, gf_dim*2], name='g_h2')))
    #h3 = tf.nn.relu(g_bn3(deconv2d(h2, [batch_size, 32, 32, gf_dim*1], name='g_h3')))
    h4 = deconv2d(h2, [batch_size, 32, 32, 3], name='g_h4')

    return tf.nn.tanh(h4)
