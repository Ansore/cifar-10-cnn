import tensorflow as tf
import input_data
import sys
from tensorflow.python.framework import graph_util
from model import base_cnn
from model import vgg16_mod
from model import official
from model import vgg19
from model import all_conv
import numpy as np


def get_loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def load_model(sess, saver,ckpt_path):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    print(latest_ckpt)
    if latest_ckpt:
        print ('resume from', latest_ckpt)
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print ('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1

def cnn_train(batch, x_train, y_train, x_test, y_test):

    num_batch = len(x_train) // batch

    print(num_batch)

    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="inputnode")
    y_ = tf.placeholder(tf.float32, [None, 10], name="classes")

    #
    #
    #
    #
    #
    # out = base_cnn.base_cnn(x)
    # out = vgg16_mod.vgg_cnn_layer(x)
    # out = all_conv.all_conv_layer(x)
    # out = official.inference(x)
    out = vgg19.vgg19_layer(x)
    #
    #
    #
    #
    #
    #
    #
    #
    #


    p = tf.nn.softmax(out, name="outnode")

    # tf.reduce_mean(p, name="outnode")

    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_prediction + np.exp(-10)), reduction_indices=[1]))

    cross_entropy = get_loss(out, y_)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('train_accuracy', accuracy)

    merged_summary_op = tf.summary.merge_all()

    # saver = tf.train.Saver()

    with tf.Session() as sess:

        # load model
        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver(tf.all_variables())
        last_epoch = load_model(sess, saver, 'save_model_tmp_vgg19/')

        # sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./tmp_vgg19', graph=tf.get_default_graph())

        for n in range(last_epoch + 1, 1000):
            # 每次取batch_size张图片
            for i in range(num_batch):
                batch_x = x_train[i*batch : (i+1)*batch]
                batch_y = y_train[i*batch : (i+1)*batch]
                # 开始训练数据，同时训练三个变量，返回三个数据

                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y})
                summary_writer.add_summary(summary, n*num_batch+i)
                # 打印损失
                if (n*num_batch+i) % 10 == 0:
                    print(n*num_batch+i, loss)

                if (n*num_batch+i) % 50 == 0:
                    # 获取测试数据的准确率
                    # test_y = sess.run([test_y])
                    x_test_t = x_test[0: 1000]
                    y_test_t = y_test[0: 1000]
                    acc = sess.run(accuracy, feed_dict={x:x_test_t, y_:y_test_t})
                    # tf.summary.scalar('test_accuracy', acc)
                    print(n*num_batch+i, acc)

                    if acc > 0.7:
                        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                                   ["outnode"])
                        with tf.gfile.FastGFile("android_vgg19/model-"+str(acc)+".pb", mode='wb') as f:
                            f.write(constant_graph.SerializeToString())

                        # saver.save(sess, './train_faces.model', global_step=n*num_batch+i)
                        # saver.save(sess, './model/train.model')
                        # sys.exit(0)

            saver.save(sess, 'save_model_tmp_vgg19/cifar.model', global_step=n)
        print('准确度小于0.81!')

if __name__ == '__main__':
    cifar10_dir = 'data/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = input_data.load_CIFAR10(cifar10_dir)
    batch_size = 128

    cnn_train(batch_size, X_train, y_train, X_test, y_test)