# _*_ coding:utf-8 _*_
#对 iris 数据集的数据进行分类
#1、数据读取
#2、训练数据的创建
#3、定义占位符X, Y 网络的数据输入
#4、定义权重weight 和偏置 bias
#5、定义一个简单的模型y=w*x+b
#6、损失函数loss
#7、图 session()
#8、训练数据
#9、验证数据valid 一个数据 目标为 [0,0,1]

import tensorflow as tf
import numpy as np
import random

def read_data(filename):
  data = []
  f=open(filename,'r')
  for line in f:
    line = line.strip('\n')
    line = line.split(',')
    num=[]
    for s in line:
      num.append(float(s))
    data.append(num)
  random.shuffle(data)
  return data

def get_batchset(batch_size, data):
  batch = len(data)/batch_size
  l=len(data)%batch_size
  x_train = []
  y_train = []
  for num in data:
    x = num[:4]
    x_train.append(x)
    if num[4] == 1.0:
      y_train.append([1., 0., 0.])
    if num[4] == 2.0:
      y_train.append([0., 1., 0.])
    if num[4] == 3.0:
      y_train.append([0., 0., 1.])
  length=len(x_train)
  x_train = np.asarray(x_train)
  x_train=x_train.reshape(length,1,4)
  y_train = np.asarray(y_train)
  y_train = y_train.reshape(length,1,3)
  return x_train, y_train

def model():
  data = read_data('iris')
  x_train, y_train = get_batchset(8, data)
  #x, y
  X = tf.placeholder(tf.float32, [1,4], name="X")
  Y = tf.placeholder(tf.float32, [1,3], name="Y")
  W = tf.Variable(tf.random_normal([4,3], stddev=0.35), name="weights")
  b = tf.Variable(tf.random_normal([3], stddev=0.5), name="bais")

  y = tf.matmul(X,W) + b

  #loss = tf.reduce_sum(tf.square(Y - y))
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y))

  optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('./path', sess.graph)
    for i in range(30):
      for xx,yy in zip(x_train, y_train):
        op, lo=sess.run([optimizer,loss], feed_dict = {X:xx, Y:yy})
      valid=np.asarray([6.2,2.9,4.3,1.3])
      valid=valid.reshape(1,4)
      print(sess.run(y,feed_dict={X:valid}))

model()
