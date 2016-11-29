# Based on https://github.com/carpedm20/DCGAN-tensorflow
import tensorflow as tf
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from network import *
from data import *

y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
batch = tf.Variable(0)
phase_train = tf.placeholder(tf.bool, name='phase_train')

batch_size = 100
sample_size = 100
image_shape = [32, 32, 3]

z_dim = 100

gf_dim = 64
df_dim = 64

learning_rate = 0.0002
beta1 = 0.5

images = tf.placeholder(tf.float32, [batch_size] + image_shape, name='real_images')
sample_images = tf.placeholder(tf.float32, [sample_size] + image_shape, name='sample_images')
z = tf.placeholder(tf.float32, [None, z_dim], name='z')

G = generator(z)
D, D_logits, predict, feat = discriminator(images)
D_, D_logits_,predict_,feat_ = discriminator(G, reuse=True)

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, y_))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))
d_loss = d_loss_real + d_loss_fake + error

g_loss = tf.nn.l2_loss(feat - feat_)/batch_size
t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

if os.path.isfile("model.ckpt"):
    saver.restore(sess, "model.ckpt")
    print("Session restored")

iterations = 700
#iterations = 200

count = 0
epoch = 40 
for k in range(epoch):
    im_id = 0
    for i in range(iterations): #epochs
        batch_images = train_X[im_id : im_id+batch_size ,:]
        labels = train_Y[im_id : im_id + batch_size ,:]
        batch_z = np.random.uniform(0, 1, [batch_size, z_dim]).astype(np.float32)
        errD_fake = d_loss_fake.eval({z: batch_z,images: batch_images}, session=sess)
        errD_real = d_loss_real.eval({images: batch_images}, session=sess)
        errG = g_loss.eval({z: batch_z,images: batch_images}, session=sess)
        errDisc = error.eval({z: batch_z, images:batch_images, y_:labels},session=sess)

        sess.run([d_optim], feed_dict={images: batch_images, z: batch_z, y_: labels})
        sess.run([g_optim], feed_dict={images: batch_images, z: batch_z, y_: labels})

        count = count + 1
        im_id = im_id + batch_size
        if i % 100 == 0: 
            print("Acc: %f" % sess.run(accuracy, feed_dict = {images: batch_images, z:batch_z, y_: labels}))
            print("Iter #: %d" % i)
            print("(Disc, Gen) error: (%f, %f)" %(errD_real + errD_fake, errG))
            print("Class error: %f" % errDisc)
    if k % 2 == 0:
        print("On iteration: " + str(k))
        G_samples = sess.run(G, feed_dict = {z : batch_z})
        save_plot(G_samples,k)
        saver.save(sess,"model.ckpt")


