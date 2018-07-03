import tensorflow as tf
import numpy as np

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

def addLayer(inputs, weight_shape, bias_shape, keep_prob, activation_function = None, name=None):
    with tf.name_scope(name):
            Weights = weight_variable(weight_shape)
            biases = bias_variable(bias_shape)
        # 卷积
            if activation_function is None:
                conv = conv2d(inputs, Weights) + biases
            else:
                conv = activation_function(conv2d(inputs, Weights) + biases)
        # 池化
            pool = max_pool_2x2(conv)
            drop = dropout(pool, keep_prob)
    return drop

def base_cnn(images):

    # layer 1 ------32->16
    drop1 = addLayer(images, [3, 3, 3, 64], [64], 1, tf.nn.relu, "layer1")

    # layer 2 -------16->8
    drop2 = addLayer(drop1, [3, 3, 64, 128], [128], 1, tf.nn.relu, "layer2")

    # layer 3 -------8->4
    drop3 = addLayer(drop2, [3, 3, 128, 256], [256], 1, tf.nn.relu, "layer3")

    # 密集连接层
    with tf.name_scope("FullyConnection"):
        Wf = weight_variable([4 * 4 * 256, 1024])
        bf = bias_variable([1024])
        drop2_flat = tf.reshape(drop3, [-1, 4 * 4 * 256])
        dense = tf.nn.relu(tf.matmul(drop2_flat, Wf) + bf)
        dropf = dropout(dense, 0.9)

    # 输出层
    with tf.name_scope("out"):
        Wout = weight_variable([1024,10])
        bout = bias_variable([10])
        out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

