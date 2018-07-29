#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from activations import SigmoidActivationFunction
from layers import FCLayer
from ffnet import FFNet
from autoencoder import Autoencoder
from check_grad_functions import check_grad, loss_func_grad

'''
python3 check_rp_forward_backward.py
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
    p = np.random.normal(size=autoencoder.net.params_number)
    Rp_loss_grad = autoencoder.compute_hessvec(p)
    hess = np.zeros((autoencoder.net.params_number))
    check_Rp_loss_grad = \
        check_grad(lambda x: loss_func_grad(autoencoder, x, inputs), w, p)
    max_diff = np.abs(Rp_loss_grad - check_Rp_loss_grad).max()
    min_diff = np.abs(Rp_loss_grad - check_Rp_loss_grad).min()
    print("compute_hessvec")
    print("min_diff =  ", min_diff)
    print("max_diff = ", max_diff)


if __name__ == '__main__':
    main()
