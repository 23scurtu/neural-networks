import numpy as np
import math
import pprint
import mnist_loader
import random
import sys


def sig(x):
    return 1/(1 + np.exp(-x))

# sig = np.vectorize(single_sig)


def sigp(x):
    return sig(x)*(1 - sig(x))

# sigp = np.vectorize(single_sigp)


class Layer:
    def __init__(self, input_size, output_size):
        self._last_inputs = None
        self._past_errors = []
        self._past_activations = []

    @property
    def last_inputs(self):
        return self._last_inputs

    @property
    def past_errors(self):
        return self._past_errors

    @property
    def past_activations(self):
        return self._past_activations

    def _propagate(self, inputs):
        raise NotImplementedError

    def _backpropagate(self, layer_error):
        raise NotImplementedError

    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        raise NotImplementedError

    def propagate(self, previous_activation, cache=False):
        inputs, activations = self._propagate(previous_activation)
        if cache:
            self._past_activations.append(activations)

        self._last_inputs = inputs

        return activations

    def backpropagate(self, layer_error, previous_inputs, cache):
        if self._last_inputs is None:
            print("Attempting to backpropagate before having propagated")

        error = self._backpropagate(layer_error, previous_inputs)
        if cache:
            self._past_errors.append(layer_error)
        return error

    def gradient_decent(self, learning_rate, previous_activations, test=False):
        self._gradient_decent(learning_rate, previous_activations, test)

        self._past_errors = []
        self._past_activations = []
        self._last_inputs = None


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, layer_size):
        self.weights = np.random.randn(layer_size, input_size)
        self.biases = np.random.randn(layer_size, 1)

        # # For tracking changes
        # self.orig_weights = self.weights.copy()
        # self.orig_biases = self.biases.copy()

        super().__init__(layer_size, input_size)

    def _propagate(self, input):
        # print(self.weights.shape)
        # print(input.shape)

        np.matmul(self.weights, input)
        layer_inputs = np.add(np.matmul(self.weights, input), self.biases)
        layer_activations = sig(layer_inputs)

        return layer_inputs, layer_activations

    def _backpropagate(self, layer_error, previous_inputs):
        # print(self.weights.transpose().shape)
        # print(layer_error.shape)

        return np.multiply(np.matmul(self.weights.transpose(), layer_error), sigp(previous_inputs))

    # Currently test is unused
    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        # print(len(previous_activations))
        # print(len(self.past_errors))

        assert len(previous_activations) == len(self.past_errors)
        minibatch_size = len(previous_activations)

        error_activations = [np.matmul(error, activation.transpose())
                             for error, activation in zip(self.past_errors, previous_activations)]

        self.weights = np.subtract(self.weights, (learning_rate / minibatch_size) * sum(error_activations))
        self.biases = np.subtract(self.biases, (learning_rate / minibatch_size) * sum(self.past_errors))


class OutputLayer(FullyConnectedLayer):
    def __init__(self, input_size, layer_size):
        super().__init__(input_size, layer_size)

    def output_error(self, actual, expected):
        error = np.multiply(self.calc_cost_partial_derivs(expected, actual), sigp(self.last_inputs))
        return error

    def calc_cost(self, expected, final_activations):
        return np.square(np.sum(np.subtract(expected - final_activations))) * 0.5

    def calc_cost_partial_derivs(self, expected, final_activations):
        return np.subtract(final_activations, expected)


class Network:
    def __init__(self, layers):
        if len(layers) < 1:
            print("Error: There must be at least 1 layers.")
            raise ValueError

        self.layers = layers
        self.learning_rate = 3

    def propagate(self, last_activation, train=False):
        for layer in self.layers:
            last_activation = layer.propagate(last_activation, train)

        return last_activation

    def backpropagate(self, input, output, actual):
        output_error = self.layers[-1].output_error(actual, output)

        last_error = output_error
        last_layer = self.layers[-1]

        for layer in reversed(self.layers[:-1]):
            # print(last_layer.weights.shape)
            # print(layer.weights.shape)
            last_error = last_layer.backpropagate(last_error, layer.last_inputs, True)
            last_layer = layer

        self.layers[0].backpropagate(last_error, input, True)

    def gradient_descent(self, minibatch_inputs, minibatch_outputs):

        assert len(minibatch_inputs) == len(minibatch_outputs)

        for inputs, output in zip(minibatch_inputs, minibatch_outputs):
            result = self.propagate(inputs, train=True)
            self.backpropagate(inputs, output, result)

        for l in range(len(self.layers) - 1, 0, -1):
            self.layers[l].gradient_decent(self.learning_rate, self.layers[l-1].past_activations)

        # TODO This has to be done after since it deletes cached activations, change caching?
        self.layers[0].gradient_decent(self.learning_rate, minibatch_inputs, True)


EPOCHS = 30
MINIBATCH_SIZE = 32


# TODO Move SGD code inside of Network
def main():
    n = Network([FullyConnectedLayer(784, 30),
                 #FullyConnectedLayer(30, 30),
                 OutputLayer(30,10)])

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    for epoch in range(EPOCHS):
        random.shuffle(training_data)

        minibatches = [training_data[k*MINIBATCH_SIZE:(k+1)*MINIBATCH_SIZE] for k in range(math.floor(len(training_data)/MINIBATCH_SIZE))]

        cnt = 0
        for minibatch in minibatches:

            minibatch_inputs = [example[0] for example in minibatch]
            minibatch_outputs = [example[1] for example in minibatch]

            n.gradient_descent(minibatch_inputs, minibatch_outputs)

            cnt += MINIBATCH_SIZE
            #print(cnt)

        TEST_COUNT = len(test_data)
        successful = 0
        for example in test_data:
            inputs = example[0]
            output = example[1]

            result = n.propagate(inputs)

            if np.argmax(result) == output:
                successful += 1

        print(successful/TEST_COUNT*100)

        # for layer in n.layers:
        #     # np.set_printoptions(threshold=sys.maxsize)
        #     print(np.subtract(layer.weights, layer.orig_weights))
        #     print('====================================================================')


if __name__ == '__main__':
    main()
