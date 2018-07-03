import tensorflow as tf
from model import vgg16_mod


def get_loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def cnn_graph():

    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="inputnode")
        y_ = tf.placeholder(tf.float32, [None, 10], name="classes")

    out = vgg16_mod.vgg_cnn_layer(x)

    p = tf.nn.softmax(out, name="outnode")

    with tf.name_scope("loss"):
        cross_entropy = get_loss(out, y_)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter('./graph_vgg16_mod', graph=sess.graph)
        sess.run(tf.global_variables_initializer())



if __name__ == '__main__':
    cnn_graph()