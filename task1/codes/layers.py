#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of layers used within neural networks

import numpy as np


class BaseLayer(object):

    def get_params_number(self):
        """
        :return num_params: number of parameters used in layer
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_weights(self):
        """
        :return w: current layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_weights(self, w):
        """
        Takes weights as a one-dimensional numpy vector and assign them to layer parameters in convenient shape,
        e.g. matrix shape for fully-connected layer
        :param w: layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_direction(self, p):
        """
        Takes direction vector as a one-dimensional numpy vector and assign it to layer parameters direction vector
        in convenient shape, e.g. matrix shape for fully-connected layer
        :param p: layer parameters direction vector, numpy vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def forward(self, inputs):
        """
        Forward propagation for layer. Intermediate results are saved within layer parameters.
        :param inputs: input batch, numpy matrix of size num_inputs x num_objects
        :return outputs: layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def backward(self, derivs):
        """
        Backward propagation for layer. Intermediate results are saved within layer parameters.
        :param derivs: loss derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_derivs: loss derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_derivs: loss derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_forward(self, Rp_inputs):
        """
        Rp forward propagation for layer. Intermediate results are saved within layer parameters.
        :param Rp_inputs: Rp input batch, numpy matrix of size num_inputs x num_objects
        :return Rp_outputs: Rp layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_backward(self, Rp_derivs):
        """
        Rp backward propagation for layer.
        :param Rp_derivs: loss Rp derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_Rp_derivs: loss Rp derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_Rp_derivs: loss Rp derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_activations(self):
        """
        :return outputs: activations computed in forward pass, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')


class FCLayer(BaseLayer):

    def __init__(self, shape, afun, use_bias=False):
        """
        :param shape: layer shape, a tuple (num_inputs, num_outputs)
        :param afun: layer activation function, instance of BaseActivationFunction
        :param use_bias: flag for using bias parameters
        """
        self.shape = shape
        self.afun = afun
        self.use_bias = use_bias

    def get_params_number(self):
        params_number = self.shape[0] * self.shape[1]
        if self.use_bias:
            params_number += self.shape[1]
        return params_number

    def get_weights(self):
        return self.w.ravel()

    def set_weights(self, w):
        if w.shape[0] != self.get_params_number():
            raise ValueError('Invalid set weights')
        if self.use_bias:
            self.w = w.reshape((self.shape[1], self.shape[0] + 1))
        else:
            self.w = w.reshape((self.shape[1], self.shape[0]))

    def set_direction(self, p):
        if p.shape[0] != self.get_params_number():
            raise ValueError('Invalid set direction')
        if self.use_bias:
            self.p = p.reshape((self.shape[1], self.shape[0] + 1))
        else:
            self.p = p.reshape((self.shape[1], self.shape[0]))

    def forward(self, inputs):
        if inputs.shape[0] != self.shape[0]:
            raise ValueError('Ivalid forward')
        if self.use_bias:
            self.inputs = np.vstack((inputs, np.ones(inputs.shape[1])))
        else:
            self.inputs = inputs
        self.u = np.dot(self.w, self.inputs)
        self.z = self.afun.val(self.u)
        return self.z

    def backward(self, derivs):

        if derivs.shape[0] != self.shape[1]:
            raise ValueError('Invalid backward')

        self.derivs = derivs
        self.u_derivs = derivs * self.afun.deriv(self.z)
        self.output_derivs = np.dot(self.w.transpose(), self.u_derivs)
        if self.use_bias:
            self.output_derivs = self.output_derivs[:-1, :]

        self.w_derivs = np.dot(self.u_derivs, self.inputs.transpose()) / \
            self.inputs.shape[1]

        return self.output_derivs, self.w_derivs.ravel()

    def Rp_forward(self, Rp_inputs):
        if Rp_inputs.shape[0] != self.shape[0]:
            raise ValueError('Invalide Rp_inputs')

        if self.use_bias:
            self.Rp_inputs = np.vstack((Rp_inputs,
                                        np.zeros(Rp_inputs.shape[1])))
        else:
            self.Rp_inputs = Rp_inputs
        self.Rp_u = np.dot(self.w, self.Rp_inputs) + np.dot(self.p, self.inputs)
        self.Rp_z = self.afun.deriv(self.z) * self.Rp_u
        return self.Rp_z

    def Rp_backward(self, Rp_derivs):
        if Rp_derivs.shape[0] != self.shape[1]:
            raise ValueError('Invalid Rp_backward')

        self.Rp_derivs = Rp_derivs
        self.Rp_u_derivs = Rp_derivs * self.afun.deriv(self.z) + \
            self.derivs * self.afun.second_deriv(self.z) * self.Rp_u

        self.Rp_output_derivs = np.dot(self.p.transpose(), self.u_derivs) + \
            np.dot(self.w.transpose(), self.Rp_u_derivs)

        if self.use_bias:
            self.Rp_output_derivs = self.Rp_output_derivs[:-1, :]

        self.Rp_w_derivs = np.dot(self.Rp_u_derivs, self.inputs.transpose()) + \
            np.dot(self.u_derivs, self.Rp_inputs.transpose())

        self.Rp_w_derivs /= self.inputs.shape[1]

        return self.Rp_output_derivs, self.Rp_w_derivs.ravel()

    def get_activations(self):
        return self.afun.val(self.u)
