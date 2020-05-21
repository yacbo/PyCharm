# -*- coding: utf-8 -*-
import tensorflow as tf

"""
#构建图
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3.,3.,3.]])
# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.],[2.]])
product = tf.matmul(matrix1,matrix2)
#在回话中启动图
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
"""



state = tf.Variable(0,name="counter")
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)
init_op = tf.initialize_all_variables()
#启动图，运行op
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
