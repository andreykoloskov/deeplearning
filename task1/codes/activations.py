#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of activation functions used within neural networks

import numpy as np


class BaseActivationFunction(object):

    def val(self, inputs):
        """
        Calculates values of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def deriv(self, inputs):
        """
        Calculates first derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def second_deriv(self, inputs):
        """
        Calculates second derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')


class LinearActivationFunction(BaseActivationFunction):
    def __init__(self):
        pass

    def val(self, inputs):
        return inputs

    def deriv(self, inputs):
        return np.ones_like(inputs)

    def second_deriv(self, inputs):
        return np.zeros_like(inputs)


class SigmoidActivationFunction(BaseActivationFunction):
    def __init__(self):
        pass

    def val(self, inputs):
        return np.ones_like(inputs) / (np.ones_like(inputs) + np.exp(-inputs))

    def deriv(self, inputs):
        return inputs * (np.ones_like(inputs) - inputs)

    def second_deriv(self, inputs):
        return inputs * (np.ones_like(inputs) - inputs) * \
                (np.ones_like(inputs) - inputs * 2.0)


class ReluActivationFunction(BaseActivationFunction):
    def __init__(self):
        pass

    def val(self, inputs):
        return np.maximum(0, inputs)

    def deriv(self, inputs):
        derivatives = np.zeros_like(inputs)
        derivatives[inputs > 0] = 1
        return derivatives

    def second_deriv(self, inputs):
        return np.zeros_like(inputs)
