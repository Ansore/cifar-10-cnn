# 修改的vgg16模型

import tensorflow as tf


# 定义各个参数随机输入,防止梯度为0
def weight_variable(shape):
    # with tf.name_scope("Weights"):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init, name="Weights")

def bias_variable(shape):
    # with tf.name_scope("biases"):
    init = tf.random_normal(shape)
    return tf.Variable(init, name="biases")

def conv2d(x, W):
    # with tf.name_scope("Conv2D"):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name="Conv2D")

def max_pool_2x2(x):
    # with tf.name_scope("MaxPool2D"):
    pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="MaxPool2D")
    return pool

def dropout(x, keep):
    # with tf.name_scope("dropout"):
    return tf.nn.dropout(x, keep, name="dropout")

#　添加神经层　
# inputs 输入数据　
# weight_shape权重格式　
# bias_shape 偏置格式
# keep_prob 过拟合
# activation_function激励函数
def add_conv(inputs, weight_shape, bias_shape, keep_prob, activation_function = None):
    Weights1 = weight_variable(weight_shape)
    biases1 = bias_variable(bias_shape)
    # 卷积
    if activation_function is None:
        conv = conv2d(inputs, Weights1) + biases1
    else:
        conv = activation_function(conv2d(inputs, Weights1) + biases1)

    drop = dropout(conv, keep_prob)
    return drop

def vgg_cnn_layer(images):
    # layer 1
    with tf.name_scope("layer1"):
        drop1 = add_conv(images, [3, 3, 3, 64], [64], 1, tf.nn.relu)
        drop1 = add_conv(drop1, [3, 3, 64, 64], [64], 1, tf.nn.relu)
        drop1 = max_pool_2x2(drop1)

    # layer 2
    with tf.name_scope("layer2"):
        drop2 = add_conv(drop1, [3, 3, 64, 128], [128], 1, tf.nn.relu)
        drop2 = add_conv(drop2, [3, 3, 128, 128], [128], 1, tf.nn.relu)
        drop2 = max_pool_2x2(drop2)

    # layer 3
    with tf.name_scope("layer3"):
        drop3 = add_conv(drop2, [3, 3, 128, 256], [256], 1, tf.nn.relu)
        drop3 = add_conv(drop3, [3, 3, 256, 256], [256], 1, tf.nn.relu)
        drop3 = max_pool_2x2(drop3)

    drop3_flat = tf.reshape(drop3, [-1, 4 * 4 * 256], name="reshape")

    dropf = dropout(drop3_flat, 0.5)

    with tf.name_scope("out"):
        Wf = weight_variable([4*4*256, 10])
        bf = bias_variable([10])
        dense = tf.matmul(dropf, Wf) + bf


    # dropf = dropout(drop2_flat, 0.5)

    # Wout = weight_variable([4*4*256, 10])
    # bout = bias_variable([10])
    #
    # out = tf.add(tf.matmul(dropf, Wout), bout)

    return dense

