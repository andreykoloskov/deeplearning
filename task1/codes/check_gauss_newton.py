#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from activations import SigmoidActivationFunction
from layers import FCLayer
from ffnet import FFNet
from autoencoder import Autoencoder
from check_grad_functions import check_grad, gauss_loss_func

'''
python3 check_gauss_newton.py
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

    p = np.random.normal(size=autoencoder.net.params_number)
    loss, loss_grad = autoencoder.compute_loss(inputs)
    gn_grad = autoencoder.compute_gaussnewtonvec(p)
    Rp_outputs = autoencoder.net.compute_Rp_outputs()
    p = np.zeros_like(w)
    check_gn_grad = np.zeros_like(w)
    num_params = autoencoder.net.params_number
    for i in range(num_params):
        p[:] = 0
        p[i] = 1
        check_gn_grad[i] = \
            check_grad(lambda x: gauss_loss_func(autoencoder, x, inputs,
                                                 Rp_outputs), w, p)
    max_diff = np.abs(gn_grad - check_gn_grad).max()
    min_diff = np.abs(gn_grad - check_gn_grad).min()

    print("compute_gauss_newton")
    print("min_diff =  ", min_diff)
    print("max_diff = ", max_diff)


if __name__ == '__main__':
    main()
