# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
X_holder = tf.placeholder(tf.float32)
y_holder = tf.placeholder(tf.float32)

def addConnect(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.01))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        return Wx_plus_b
    else:
        return activation_function(Wx_plus_b)

connect_1 = addConnect(X_holder, 784, 300, tf.nn.relu)
predict_y = addConnect(connect_1, 300, 10, tf.nn.softmax)
loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
optimizer = tf.train.AdagradOptimizer(0.3)
train = optimizer.minimize(loss)

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

for i in range(1000):
    images, labels = mnist.train.next_batch(batch_size)
    session.run(train, feed_dict={X_holder:images, y_holder:labels})
    if i % 50 == 0:
        correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = session.run(accuracy, feed_dict={X_holder:mnist.test.images, y_holder:mnist.test.labels})
        print('step:%d accuracy:%.4f' %(i, accuracy_value))