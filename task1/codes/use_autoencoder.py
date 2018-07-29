#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import math
import argparse
from sklearn.datasets.mldata import fetch_mldata
from matplotlib import pyplot as plt
from activations import SigmoidActivationFunction, LinearActivationFunction
from layers import FCLayer
from ffnet import FFNet
from autoencoder import Autoencoder

'''
python3 use_autoencoder.py -p <path to MNIST folder>
         -t <type of gradient function> -tr <train data size>
         -ts <test data size> -e <number of epochs> -ms <minibatch size>
         -mm <momentum> -d <print to display>
'''


def main():
    '''
    use of an autocoder
    param path: path to folder where you are loading MNIST
    param type: type of gradient function (sgd, sgd_momentum, rmsprop, adam)
    param train_size: train data size
    param test_size: test data size
    param num_epoch: number of epochs
    param minibatch_size: minibatch size
    param momentum: momentum
    param display: print to display
    '''
    options = parse_args()
    mnist = fetch_mldata('MNIST original', data_home=options['path'])
    data = mnist.data.astype('float64')

    train_size = options['train_size']
    train_data = data[np.random.choice(data.shape[0], train_size, False), :]
    test_size = options['test_size']
    test_data = data[np.random.choice(data.shape[0], test_size, False), :]

    autoencoder = Autoencoder([
            FCLayer((784, 250), SigmoidActivationFunction(), True),
            FCLayer((250, 50), SigmoidActivationFunction(), True),
            FCLayer((50, 2), SigmoidActivationFunction(), True),
            FCLayer((2, 50), LinearActivationFunction(), True),
            FCLayer((50, 250), SigmoidActivationFunction(), True),
            FCLayer((250, 784), SigmoidActivationFunction(), True)])

    if options['type'] == 'sgd':
        res = autoencoder.run_sgd(train_data.transpose(), step_size=1.0,
                                  momentum=0, num_epoch=options['num_epoch'],
                                  minibatch_size=options['minibatch_size'],
                                  l2_coef=1e-4,
                                  test_inputs=test_data.transpose(),
                                  display=options['display'])
    elif options['type'] == 'sgd_momentum':
        res = autoencoder.run_sgd(train_data.transpose(), step_size=1.0,
                                  momentum=options['momentum'],
                                  num_epoch=options['num_epoch'],
                                  minibatch_size=options['minibatch_size'],
                                  l2_coef=1e-4,
                                  test_inputs=test_data.transpose(),
                                  display=options['display'])
    elif options['type'] == 'rmsprop':
        res = autoencoder.run_rmsprop(train_data.transpose(), step_size=1.0,
                                      num_epoch=options['num_epoch'],
                                      minibatch_size=options['minibatch_size'],
                                      l2_coef=1e-4,
                                      test_inputs=test_data.transpose(),
                                      display=options['display'])
    elif options['type'] == 'adam':
        res = autoencoder.run_adam(train_data.transpose(), step_size=1.0,
                                   num_epoch=options['num_epoch'],
                                   minibatch_size=options['minibatch_size'],
                                   l2_coef=1e-4,
                                   test_inputs=test_data.transpose(),
                                   display=options['display'])

    print(res)

    plt.title('test loss')
    plt.scatter(np.arange(len(res['test_loss'])), res['test_loss'])
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        help='path to folder where you are loading MNIST')
    parser.add_argument('--type', '-t', type=str,
                        help='type of gradient function')
    parser.add_argument('--train_size', '-tr', type=int, help='train data size')
    parser.add_argument('--test_size', '-ts', type=int, help='test data size')
    parser.add_argument('--num_epoch', '-e', type=int, help='number of epochs')
    parser.add_argument('--minibatch_size', '-ms', type=int,
                        help='minibatch size')
    parser.add_argument('--momentum', '-mm', type=float, help='momentum')
    parser.add_argument('--display', '-d', type=bool, help='print to display')
    options = parser.parse_args()
    return vars(options)


if __name__ == '__main__':
    main()
