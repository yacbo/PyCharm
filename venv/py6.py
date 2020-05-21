# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#数据准备
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
X_holder = tf.placeholder(tf.float32)
y_holder = tf.placeholder(tf.float32)


def addConnect(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.01)) #定义权重矩阵
    biases = tf.Variable(tf.zeros([1, out_size])) #定义偏置矩阵
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

init = tf.global_variables_initializer()
session = tf.Session()  #实例化会话对象
session.run(init)  #变量初始化

#模型迭代训练1000次,每隔50次训练打印模型准确率
for i in range(2000):
    images, labels = mnist.train.next_batch(batch_size)
    session.run(train, feed_dict={X_holder:images, y_holder:labels})
    if i % 50 == 0:
        correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = session.run(accuracy, feed_dict={X_holder:mnist.test.images, y_holder:mnist.test.labels})
        print('step:%d accuracy:%.4f' %(i, accuracy_value))

def drawDigit2(position, image, title, isTrue):
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap='gray_r')
    plt.axis('off')
    if not isTrue:
        plt.title(title, color='red')
    else:
        plt.title(title)

#模型测试
def batchDraw2(batch_size):
    images, labels = mnist.test.next_batch(batch_size)
    predict_labels = session.run(predict_y, feed_dict={X_holder: images, y_holder: labels})
    image_number = images.shape[0]
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number
    plt.figure(figsize=(row_number + 8, column_number + 8))
    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index + 1)
                image = images[index]
                actual = np.argmax(labels[index])
                predict = np.argmax(predict_labels[index])
                isTrue = actual == predict
                title = 'actual:%d\npredict:%d' % (actual, predict)
                drawDigit2(position, image, title, isTrue)
batchDraw2(120)
plt.show()