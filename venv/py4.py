# -*- coding: utf-8 -*-

"""
import tensorflow as tf
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
"""
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt  #画图

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

image = mnist.train.images[2].reshape(-1,28)
plt.subplot(131)
plt.imshow(image)
plt.axis('off')
plt.subplot(132)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(133)
plt.imshow(image, cmap='gray_r')
plt.axis('off')
plt.show()