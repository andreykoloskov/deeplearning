#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from activations import SigmoidActivationFunction
from layers import FCLayer
from ffnet import FFNet
from autoencoder import Autoencoder
from check_grad_functions import check_grad, loss_func

'''
python3 check_forward_backward.py
'''


def main():
    inputs = np.random.random((5, 5))
    autoencoder = Autoencoder([
            FCLayer((5, 4), SigmoidActivationFunction(), True),
            FCLayer((4, 3), SigmoidActivationFunction(), True),
            FCLayer((3, 4), SigmoidActivationFunction(), True),
            FCLayer((4, 5), SigmoidActivationFunction(), True)])

    w = np.random.normal(size=autoencoder.net.params_number)
    autoencoder.net.set_weights(w)
    loss, loss_grad = autoencoder.compute_loss(inputs)
    num_params = autoencoder.net.params_number
    p = np.zeros((autoencoder.net.params_number))
    check_loss_grad = np.zeros((autoencoder.net.params_number))
    for i in range(num_params):
        p[:] = 0
        p[i] = 1
        check_loss_grad[i] = \
            check_grad(lambda x: loss_func(autoencoder, x, inputs), w, p)
    max_diff = np.abs(loss_grad - check_loss_grad).max()
    min_diff = np.abs(loss_grad - check_loss_grad).min()
    print("compute_loss")
    print("min_diff =  ", min_diff)
    print("max_diff = ", max_diff)

if __name__ == '__main__':
    main()
