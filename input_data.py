# -*- coding: utf-8 -*-
import pickle as p
import numpy as np
import os


def load_CIFAR_batch(filename):
    """ 载入cifar数据集的一个batch """
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
        Y = np.array(Y)
        return X, Y

# one hot 处理
def make_one_hot(data1):
    return (np.arange(10)==data1[:,None]).astype(np.integer)


def load_CIFAR10(ROOT):
    """ 载入cifar全部数据 """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    # one hot 处理
    Ytr = make_one_hot(Ytr)
    Yte = make_one_hot(Yte)

    return Xtr, Ytr, Xte, Yte

if __name__ == '__main__':
    # 载入CIFAR-10数据集
    cifar10_dir = 'data/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # 看看数据集中的一些样本：每个类别展示一些
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)