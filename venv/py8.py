# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# creat tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 0.5表示学习效率
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# create tensorflow structure end

sess = tf.Session();
sess.run(init);  # Very important

for step in range(10001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases));

"""
 #session的两种打开模式(1)
 sess = tf.Session()
 sess.run(product)

#session的两种打开模式(2)
with tf.Session() as sess:
    result2 = sess.run(product)
print (result2)
"""