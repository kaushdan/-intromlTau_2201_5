"""

backprop_network.py

"""


import random

import numpy as np

from scipy.special import softmax

import math


class Network(object):
    def __init__(self, sizes):

        """The list ``sizes`` contains the number of neurons in the

        respective layers of the network.  For example, if the list

        was [2, 3, 1] then it would be a three-layer network, with the

        first layer containing 2 neurons, the second layer 3 neurons,

        and the third layer 1 neuron.  The biases and weights for the

        network are initialized randomly, using a Gaussian

        distribution with mean 0, and variance 1.  Note that the first

        layer is assumed to be an input layer, and by convention we

        won't set any biases for those neurons, since biases are only

        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        learning_rate,
        test_data,
        calc_stats=False,
    ):

        """Train the neural network using mini-batch stochastic

        gradient descent.  The ``training_data`` is a list of tuples

        ``(x, y)`` representing the training inputs and the desired

        outputs."""
        stats = []
        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))

        n = len(training_data)

        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:

                self.update_mini_batch(mini_batch, learning_rate)

            print(
                "Epoch {0} test accuracy: {1}".format(
                    j, self.one_label_accuracy(test_data)
                )
            )
            if calc_stats:
                stats.append(
                    [
                        self.one_label_accuracy(test_data),
                        self.one_hot_accuracy(training_data),
                        self.loss(training_data),
                    ]
                )

        return np.array(stats)

    def update_mini_batch(self, mini_batch, learning_rate):

        """Update the network's weights and biases by applying

        stochastic gradient descent using backpropagation to a single mini batch.

        The ``mini_batch`` is a list of tuples ``(x, y)``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - (learning_rate / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]

        self.biases = [
            b - (learning_rate / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """The function receives as input a 784 dimensional

        vector x and a one-hot vector y.

        The function should return a tuple of two lists (db, dw)

        as described in the assignment pdf."""

        # Your backprop implementation.
        z = [None for _ in range(self.num_layers)]
        v = [None for _ in range(self.num_layers)]
        z[0] = x
        layer = 0

        for idx, (b, w) in enumerate(zip(self.biases, self.weights), start=1):
            v[idx] = w @ z[idx - 1] + b
            if layer == len(self.weights) - 1:
                z[idx] = softmax(v[idx])

            else:
                z[idx] = relu(v[idx])

            layer += 1

        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta = z[len(self.weights)] - y
        delta_nabla_w[len(self.weights) - 1] = (z[len(self.weights)] - y) @ z[
            len(self.weights) - 1
        ].transpose()

        delta_nabla_b[len(self.weights) - 1] = z[len(self.weights)] - y

        for layer, (b, w) in reversed(
            list(enumerate(zip(self.biases[1:], self.weights[1:]), start=1))
        ):
            if layer == len(self.weights) - 1:
                delta = w.transpose() @ delta
                delta_nabla_w[layer - 1] = (delta * relu_derivative(v[layer])) @ z[
                    layer - 1
                ].transpose()

                delta_nabla_b[layer - 1] = delta * relu_derivative(v[layer])

            else:
                delta = w.transpose() @ (relu_derivative(v[layer + 1]) * delta)

                delta_nabla_w[layer - 1] = (delta * relu_derivative(v[layer])) @ z[
                    layer - 1
                ].transpose()
                delta_nabla_b[layer - 1] = delta * relu_derivative(v[layer])

        return delta_nabla_b, delta_nabla_w

    def one_label_accuracy(self, data):

        """Return accuracy of network on data with numeric labels"""

        output_results = [
            (np.argmax(self.network_output_before_softmax(x)), y) for (x, y) in data
        ]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))

    def one_hot_accuracy(self, data):

        """Return accuracy of network on data with one-hot labels"""

        output_results = [
            (np.argmax(self.network_output_before_softmax(x)), np.argmax(y))
            for (x, y) in data
        ]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))

    def network_output_before_softmax(self, x):

        """Return the output of the network before softmax if ``x`` is input."""

        layer = 0

        for b, w in zip(self.biases, self.weights):

            if layer == len(self.weights) - 1:

                x = np.dot(w, x) + b

            else:

                x = relu(np.dot(w, x) + b)

            layer += 1

        return x

    def loss(self, data):

        """Return the CE loss of the network on the data"""

        loss_list = []

        for (x, y) in data:

            net_output_before_softmax = self.network_output_before_softmax(x)

            net_output_after_softmax = self.output_softmax(net_output_before_softmax)

            loss_list.append(
                np.dot(-np.log(net_output_after_softmax).transpose(), y).flatten()[0]
            )

        return sum(loss_list) / float(len(data))

    def output_softmax(self, output_activations):

        """Return output after softmax given output before softmax"""

        return softmax(output_activations)

    def loss_derivative_wr_output_activations(self, output_activations, y):

        # Implement derivative of loss with respect to the output activations before softmax

        return self.output_softmax(output_activations) - y


def relu(z):
    """Implement the relu function."""

    return np.maximum(0, z)


def relu_derivative(z):
    """Implement the derivative of the relu function."""
    return 1.0 * (z > 0)
