#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def check_grad(func, t, p, e=1e-08):
    """
    Check gradient for function
    :param func: callable
    :param t: numpy array
    :param p: numpy array
    :param e: float
    :return: gradient
    """
    return (func(t + e * p) - func(t)) / e


def loss_func(autoencoder, weights, inputs):
    '''
    Loss function
    :param autoencoder
    :param weights numpy array
    :param imputs numpy array
    :return loss
    '''
    autoencoder.net.set_weights(weights)
    outputs = autoencoder.net.compute_outputs(inputs)
    return 0.5 * np.sum((inputs - outputs) ** 2, axis=0).mean()


def loss_func_grad(autoencoder, weights, inputs):
    '''
    Gradient function
    :param autoencoder
    :param weights numpy array
    :param imputs numpy array
    :return gradient
    '''
    autoencoder.net.set_weights(weights)
    outputs = autoencoder.net.compute_outputs(inputs)
    outputs, grad = autoencoder.compute_loss(inputs)
    return grad


def gauss_loss_func(autoencoder, weights, inputs, direction):
    '''
    Gauss loss function
    :param autoencoder
    :param weights numpy array
    :param imputs numpy array
    :param directrion numpy array
    :return gauss loss
    '''
    autoencoder.net.set_weights(weights)
    outputs = autoencoder.net.compute_outputs(inputs)
    return np.sum(outputs * direction, axis=0).mean()
