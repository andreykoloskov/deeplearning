#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of autoencoder using general feed-forward neural network

import numpy as np
import math
import ffnet
import matplotlib.pylab as plt


class Autoencoder:

    def __init__(self, layers):
        """
        :param layers: a list of fully-connected layers
        """
        self.net = ffnet.FFNet(layers)

        if self.net.layers[0].shape[0] != self.net.layers[-1].shape[1]:
            raise ValueError('In the given autoencoder number of inputs and outputs is different!')

    def compute_loss(self, inputs):
        """
        Computes autoencoder loss value and loss gradient using given batch of data
        :param inputs: numpy matrix of size num_features x num_objects
        :return loss: loss value, a number
        :return loss_grad: loss gradient, numpy vector of length num_params
        """
        outputs = self.net.compute_outputs(inputs)
        loss_grad = self.net.compute_loss_grad(outputs - inputs)
        loss = np.sum((inputs - outputs) ** 2, axis=0).mean() / 2.0
        return loss, loss_grad

    def compute_hessvec(self, p):
        """
        Computes a product of Hessian and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Hp: a numpy vector of length num_params
        """
        self.net.set_direction(p)
        Rp_outputs = self.net.compute_Rp_outputs()
        return self.net.compute_loss_Rp_grad(Rp_outputs)

    def compute_gaussnewtonvec(self, p):
        """
        Computes a product of Gauss-Newton Hessian approximation and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Gp: a numpy vector of length num_params
        """
        self.net.set_direction(p)
        Rp_outputs = self.net.compute_Rp_outputs()
        return self.net.compute_loss_grad(Rp_outputs)

    def run_sgd(self, inputs, step_size=0.01, momentum=0.9, num_epoch=200,
                minibatch_size=100, l2_coef=1e-5, test_inputs=None,
                display=False):
        """
        Stochastic gradient descent optimization
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param momentum: momentum coefficient, number
        :param num_epoch: number of training epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs, the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each peoch, list
        """
        train_loss = []
        train_grad = []
        test_loss = []
        test_grad = []
        max_inputs = np.max(inputs) + 1.0
        max_test_inputs = np.max(test_inputs) + 1.0
        inputs = inputs / max_inputs
        test_inputs = test_inputs / max_test_inputs
        w = np.random.normal(loc=0.0, scale=0.2, size=self.net.params_number)
        self.net.set_weights(w)
        for i in range(num_epoch):
            first = 0
            dw = 0
            while first < inputs.shape[1]:
                step_input = inputs[:, first: first + minibatch_size]
                first += minibatch_size
                step_output = self.net.compute_outputs(step_input)
                loss = np.sum(np.abs(step_output - step_input), axis=0).mean()
                dz = np.sign(step_output - step_input)
                loss_grad = self.net.compute_loss_grad(dz)
                w = self.net.get_weights()
                l2_loss_grad = loss_grad + 2.0 * l2_coef * w
                learning_rate = step_size / (i + 1) ** 0.25
                dw = momentum * dw - learning_rate * l2_loss_grad
                w = w + dw
                self.net.set_weights(w)
                if first >= inputs.shape[1]:
                    loss_grad_norm = math.sqrt(np.sum(l2_loss_grad ** 2))
                    train_loss.append(loss)
                    train_grad.append(loss_grad_norm)

                    test_outputs = self.net.compute_outputs(test_inputs)
                    loss2 = np.sum(np.abs(test_outputs - test_inputs),
                                   axis=0).mean()
                    dz = np.sign(test_outputs - test_inputs)
                    test_loss_grad = self.net.compute_loss_grad(dz)
                    test_loss_grad_norm = math.sqrt(np.sum(test_loss_grad ** 2))
                    test_loss.append(loss2)
                    test_grad.append(test_loss_grad_norm)

                    if display is True:
                        print(loss)

        return {'train_loss': train_loss, 'train_grad': train_grad,
                'test_loss': test_loss, 'test_grad': test_grad}

    def run_rmsprop(self, inputs, step_size=0.01, num_epoch=200,
                    minibatch_size=100, l2_coef=1e-5, test_inputs=None,
                    display=False):
        """
        RMSprop stochastic optimization
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param num_epoch: number of training epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs, the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each peoch, list
        """
        train_loss = []
        train_grad = []
        test_loss = []
        test_grad = []
        max_inputs = np.max(inputs) + 1.0
        max_test_inputs = np.max(test_inputs) + 1.0
        inputs = inputs / max_inputs
        test_inputs = test_inputs / max_test_inputs
        w = np.random.normal(loc=0.0, scale=0.2, size=self.net.params_number)
        self.net.set_weights(w)
        for i in range(num_epoch):
            first = 0
            v = 0
            while first < inputs.shape[1]:
                step_input = inputs[:, first: first + minibatch_size]
                first += minibatch_size
                step_output = self.net.compute_outputs(step_input)
                loss = np.sum(np.abs(step_output - step_input), axis=0).mean()
                dz = np.sign(step_output - step_input)
                loss_grad = self.net.compute_loss_grad(dz)
                w = self.net.get_weights()
                l2_loss_grad = loss_grad + 2.0 * l2_coef * w
                v = 0.9 * v + 0.1 * np.dot(l2_loss_grad, l2_loss_grad)
                learning_rate = step_size / (i + 1) ** 0.25
                w = w - learning_rate * l2_loss_grad / np.sqrt(v)
                self.net.set_weights(w)
                if first >= inputs.shape[1]:
                    loss_grad_norm = math.sqrt(np.sum(l2_loss_grad ** 2))
                    train_loss.append(loss)
                    train_grad.append(loss_grad_norm)

                    test_outputs = self.net.compute_outputs(test_inputs)
                    loss2 = np.sum(np.abs(test_outputs - test_inputs),
                                   axis=0).mean()
                    dz = np.sign(test_outputs - test_inputs)
                    test_loss_grad = self.net.compute_loss_grad(dz)
                    test_loss_grad_norm = math.sqrt(np.sum(test_loss_grad ** 2))
                    test_loss.append(loss2)
                    test_grad.append(test_loss_grad_norm)

                    if display is True:
                        print(loss)

        return {'train_loss': train_loss, 'train_grad': train_grad,
                'test_loss': test_loss, 'test_grad': test_grad}

    def run_adam(self, inputs, step_size=0.01, num_epoch=200,
                 minibatch_size=100, l2_coef=1e-5, test_inputs=None,
                 display=False):
        """
        ADAM stochastic optimization
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param num_epoch: maximal number of epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs, the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each peoch, list
        """
        train_loss = []
        train_grad = []
        test_loss = []
        test_grad = []
        max_inputs = np.max(inputs) + 1.0
        max_test_inputs = np.max(test_inputs) + 1.0
        inputs = inputs / max_inputs
        test_inputs = test_inputs / max_test_inputs
        w = np.random.normal(loc=0.0, scale=0.2, size=self.net.params_number)
        self.net.set_weights(w)
        for i in range(num_epoch):
            first = 0
            m = 0
            v = 0
            b1 = 0.9
            b2 = 0.999
            e = 1e-10
            while first < inputs.shape[1]:
                step_input = inputs[:, first: first + minibatch_size]
                first += minibatch_size
                step_output = self.net.compute_outputs(step_input)
                loss = np.sum(np.abs(step_output - step_input), axis=0).mean()
                dz = np.sign(step_output - step_input)
                loss_grad = self.net.compute_loss_grad(dz)
                w = self.net.get_weights()
                l2_loss_grad = loss_grad + 2.0 * l2_coef * w
                b1_ = b1 ** (i + 1)
                b2_ = b2 ** (i + 1)
                m = b1 * m + (1.0 - b1) * l2_loss_grad
                v = b2 * v + (1.0 - b2) * np.dot(l2_loss_grad, l2_loss_grad)
                m_ = m / (1.0 - b1_)
                v_ = v / (1.0 - b2_)
                learning_rate = step_size / (i + 1) ** 0.25
                w = w - learning_rate * m_ / (np.sqrt(v_) + e)
                self.net.set_weights(w)
                if first >= inputs.shape[1]:
                    loss_grad_norm = math.sqrt(np.sum(l2_loss_grad ** 2))
                    train_loss.append(loss)
                    train_grad.append(loss_grad_norm)

                    test_outputs = self.net.compute_outputs(test_inputs)
                    loss2 = np.sum(np.abs(test_outputs - test_inputs),
                                   axis=0).mean()
                    dz = np.sign(test_outputs - test_inputs)
                    test_loss_grad = self.net.compute_loss_grad(dz)
                    test_loss_grad_norm = math.sqrt(np.sum(test_loss_grad ** 2))
                    test_loss.append(loss2)
                    test_grad.append(test_loss_grad_norm)

                    if display is True:
                        print(loss)

        return {'train_loss': train_loss, 'train_grad': train_grad,
                'test_loss': test_loss, 'test_grad': test_grad}
