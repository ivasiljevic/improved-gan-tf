import tensorflow as tf
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from svhn_model import *
from svhn_data import *
from scipy.ndimage.filters import gaussian_filter

y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
batch = tf.Variable(0)
phase_train = tf.placeholder(tf.bool, name='phase_train')
###############################################################################

batch_size = 32
image_size = 108
sample_size = 32
image_shape = [32, 32, 3]

z_dim = 100

gf_dim = 64
df_dim = 64

learning_rate = 0.0003
beta1 = 0.5

images = tf.placeholder(tf.float32, [batch_size] + image_shape, name='real_images')
sample_images= tf.placeholder(tf.float32, [sample_size] + image_shape, name='sample_images')
z = tf.placeholder(tf.float32, [None, z_dim], name='z')

G = generator(z)
D, D_logits, predict = discriminator(images)
D_, D_logits_,predict_ = discriminator(G, reuse=True)

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, y_))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))
d_loss = d_loss_real + d_loss_fake + error

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_)))

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
iterations = 2200

count = 0
epoch = 20 
for k in range(epoch):
    im_id = 0
    for i in range(iterations): #epochs
        batch_images = train_X[im_id : im_id+batch_size ,:]
        labels = train_Y[im_id : im_id + batch_size ,:]
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

        # Update D network
        sess.run([d_optim], feed_dict={images: batch_images, z: batch_z, y_: labels})
        # Update G network
        #sess.run([g_optim], feed_dict={z: batch_z})
        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        sess.run([g_optim], feed_dict={z: batch_z})
        errD_fake = d_loss_fake.eval({z: batch_z}, session=sess)
        errD_real = d_loss_real.eval({images: batch_images}, session=sess)
        errG = g_loss.eval({z: batch_z}, session=sess)
        #print("Error of D: %.8f",errD_real+errD_fake)
        count = count + 1
        im_id = im_id + batch_size
        if count % 1000 == 0: 
            print("Acc: ",sess.run(accuracy, feed_dict = {images: batch_images, z:batch_z, y_: labels}))
            print("Image #: ", count)
    if k % 5 == 0:
        print("On iteration: " + str(k))
        G_samples = sess.run(G, feed_dict = {z : batch_z})
        save_plot(G_samples,k)
        saver.save(sess,"model.ckpt")


