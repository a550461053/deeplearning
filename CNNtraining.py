# coding=gbk
'''
Created on 2017年5月2日

@author: yuziqi
'''
from __future__ import print_function
import tensorflow as tf
# import numpy as np
import matplotlib.pyplot as plt

# 设置按需使用GPU
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.InteractiveSession(config=config)

from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


# add one more layer and return the output of this layer
# def add_layer(inputs, in_size, out_size, activation_function=None):
#     with tf.name_scope('layer'):
#         with tf.name_scope('weigths'):
#             Weights = tf.Variable(tf.random_normal([in_size, out_size]))
#         with tf.name_scope('biases'):
#             biases = tf.Variable(tf.zeros([1, out_size]) + 0,1)
#         with tf.name_scope('Wx_plus_b'):
#             Wx_plus_b = tf.matmul(inputs, Weights) + biases
#         if activation_function is None:
#             outputs = Wx_plus_b
#         else:
#             outputs = activation_function(Wx_plus_b)
#         return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob: 1.0})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs,ys:v_ys, keep_prob: 1.0})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[4] = 1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

######################### 1. 定义网络的输入输出 #######################

# None是样本数目，表示多少输入数据都行，784是输入数据的特征数目
xs = tf.placeholder(tf.float32, [None, 784])/255. # 28*28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28,28,1])
# print(x_image.shape) # [n_samples, 28,28,1]


######################### 2. 定义网络层 #######################

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32]) # patch 5*5,in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # patch 5*5,in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                          # output size 7*7*64

## func1 layer ##
print("func1 layer ...")
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7,7,64]->>[n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
print("func2 layer ...")
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

######################### 3. 求解神经网络参数 #######################
# 3.1 定义损失函数：the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

# train_step = tf.train_and_test.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 3.2 定义训练过程：也就是选择optimizer来使loss达到最小
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 3.3 变量初始化
sess = tf.InteractiveSession()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
print(tf.__version__)
# init = tf.initialize_all_variables()
sess.run(init)


# # 3.4 表示中间变量
# ### 输入数据图像基本操作
# img1 = mnist.train_and_test.images[1]
# label1 = mnist.train_and_test.labels[1]
# print(label1)
# print("image shape= ", (img1.shape))
# img1.shape = [28, 28]
#
# # 显示了热度图像
# plt.imshow(img1)
# plt.axis('off') # 不显示坐标轴
# plt.show()
#
# # 显示灰度图像
# plt.imshow(img1, cmap='gray')
# plt.show()
#
# # 显示在一个图上
# plt.subplot(4,8,1)
# plt.imshow(img1, cmap='gray')
# plt.axis('off')
# plt.subplot(4,8,2)
# plt.imshow(img1, cmap='gray')
# plt.axis('off')
# plt.show()
#
# ### 显示网络中间结果
#
# # 首先应该把 img1 转为正确的shape (None, 784)
# X_img = img1.reshape([-1, 784])
# y_img = mnist.train_and_test.labels[1].reshape([-1, 10])
# # 我们要看 Conv1 的结果，即 h_conv1
# result = h_conv1.eval(feed_dict={xs: X_img, ys: y_img, keep_prob: 1.0})
# print(result.shape)
# print(type(result))
#
# for _ in range(32):
#     show_img = result[:,:,:,_]
#     show_img.shape = [28, 28]
#     plt.subplot(4, 8, _ + 1)
#     plt.imshow(show_img, cmap='gray')
#     plt.axis('off')
# plt.show()
#
#
# # 首先应该把 img1 转为正确的shape (None, 784)
# X_img = mnist.train_and_test.images[2].reshape([-1, 784])
# y_img = mnist.train_and_test.labels[1].reshape([-1, 10]) # 这个标签只要维度一致就行了
# result = h_conv1.eval(feed_dict={xs: X_img, ys: y_img, keep_prob: 1.0})
#
# for _ in range(32):
#     show_img = result[:,:,:,_]
#     show_img.shape = [28, 28]
#     plt.subplot(4, 8, _ + 1)
#     plt.imshow(show_img, cmap='gray')
#     plt.axis('off')
# plt.show()


# 3.5 开始train
print("training ...")
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})

    if i % 50 == 0:
#         print(shape(mnist.test.images), shape(mnist.test.labels))
#         print(mnist.test.images.shape)
        # 直接观察loss； 或者用训练测试集
        print('loss:',sess.run(cross_entropy, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5}))
        result = compute_accuracy(batch_xs, batch_ys)
        print("step:%d" % i,", acc:%g" %result)
sess.close()

print("train_and_test finished!")
## 如果一次性来做测试的话，可能占用的显存会比较多，所以测试的时候也可以设置较小的batch来看准确率


######################### 4. 测试模型 #######################
# print("testing ...")
# rr = 0
# for i in range(100):
#     x_batch, y_batch = mnist.test.next_batch(50)
#     acc = (compute_accuracy(x_batch, y_batch))
#     print("step:%d" %i,", acc:%g" %acc)
#     rr = rr + acc
# print("评价：%g -- test finished!" % (rr/100.0))




