import tensorflow as tf

# 定义各个参数随机输入,防止梯度为0
def weight_variable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return pool

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

#　添加卷积层
# inputs 输入数据　
# weight_shape权重格式　
# bias_shape 偏置格式
# keep_prob 过拟合
# activation_function激励函数
def addLayer(inputs, weight_shape, bias_shape, keep_prob, activation_function = None):
    Weights1 = weight_variable(weight_shape)
    biases1 = bias_variable(bias_shape)
    # 卷积
    if activation_function is None:
        conv = conv2d(inputs, Weights1) + biases1
    else:
        conv = activation_function(conv2d(inputs, Weights1) + biases1)
    drop = dropout(conv, keep_prob)
    return drop

def all_conv_layer(images):

    # # 2315 步 最高 57%
    # # layer 1
    # drop1 = addLayer(images, [3, 3, 3, 96], [96], 1, tf.nn.relu)
    # drop1 = addLayer(drop1, [3, 3, 96, 96], [96], 1, tf.nn.relu)
    # drop1 = max_pool_2x2(drop1)
    #
    # # layer 2
    # drop2 = addLayer(drop1, [3, 3, 96, 192], [192], 1, tf.nn.relu)
    # drop2 = addLayer(drop2, [3, 3, 192, 192], [192], 1, tf.nn.relu)
    # drop2 = max_pool_2x2(drop2)
    #
    # drop3 = addLayer(drop2, [3, 3, 192, 192], [192], 1, tf.nn.relu)
    #
    # drop4 = addLayer(drop3, [1, 1, 192, 192], [192], 1, tf.nn.relu)
    #
    # drop5 = addLayer(drop4, [1, 1, 192, 10], [10], 1, tf.nn.relu)
    #
    # drop2_flat = tf.reshape(drop5, [-1, 8 * 8 * 10])
    #
    # Wout = weight_variable([8 * 8 * 10, 10])
    # bout = bias_variable([10])
    # out = tf.add(tf.matmul(drop2_flat, Wout), bout)

    # layer 1
    drop1 = addLayer(images, [3, 3, 3, 96], [96], 1, tf.nn.relu)
    drop1 = addLayer(drop1, [3, 3, 96, 96], [96], 1, tf.nn.relu)
    drop1 = addLayer(drop1, [3, 3, 96, 96], [96], 1, tf.nn.relu)
    drop1 = max_pool_2x2(drop1)

    # layer 2
    drop2 = addLayer(drop1, [3, 3, 96, 192], [192], 1, tf.nn.relu)
    drop2 = addLayer(drop2, [3, 3, 192, 192], [192], 1, tf.nn.relu)
    drop2 = addLayer(drop2, [3, 3, 192, 192], [192], 1, tf.nn.relu)
    drop2 = max_pool_2x2(drop2)

    drop2_flat = tf.reshape(drop2, [-1, 8 * 8 * 192])

    Wout = weight_variable([8 * 8 * 192, 10])
    bout = bias_variable([10])
    out = tf.add(tf.matmul(drop2_flat, Wout), bout)

    return out