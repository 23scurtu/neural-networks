import numpy as np
import math
import pprint
import mnist_loader
import random
import sys
import timeit
import time
from im2col import *
import pickle
import argparse


def sig(x):
    return 1/(1 + np.exp(-x))

# sig = np.vectorize(single_sig)


def sigp(x):
    return sig(x)*(1 - sig(x))

# sigp = np.vectorize(single_sigp)


def relu(x):
    # return np.maximum(0, x)
    y1 = ((x > 0) * x)
    y2 = ((x <= 0) * x * 0.01)
    return y1 + y2


def relup(x):
    r = np.copy(x)
    r[r <= 0] = 0.01
    r[r > 0] = 1
    return r


class Layer:
    def __init__(self, input_size, output_size, activation_function=None, activation_derivative=None):
        self._last_inputs = None
        self._past_errors = []
        self._past_activations = []

        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    @property
    def last_inputs(self):
        # weighted input to the neurons in this layer, prior to activation function.
        # TODO Rename to last_weighted_inputs?
        return self._last_inputs

    @property
    def past_errors(self):
        return self._past_errors

    @property
    def past_activations(self):
        return self._past_activations

    def _propagate(self, inputs):
        raise NotImplementedError

    def _backpropagate(self, layer_error, previous_inputs):
        raise NotImplementedError

    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        raise NotImplementedError

    def propagate(self, previous_activation, cache=False):
        layer_inputs = self._propagate(previous_activation)
        activations = (self.activation_function(layer_inputs) if self.activation_function else layer_inputs)

        if cache:
            self._past_activations.append(activations)

        self._last_inputs = layer_inputs

        return activations

    def backpropagate(self, layer_error, previous_inputs, cache):
        # TODO Since activation function refactor, previous_inputs are no longer used.
        if self._last_inputs is None:
            print("Attempting to backpropagate before having propagated")

        # Layer error passed in is not yet w.r.t to layer inputs
        if self.activation_derivative:
            layer_error = np.multiply(layer_error, self.activation_derivative(self.last_inputs))

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
    def __init__(self, input_size, layer_size, activation_function=sig, activation_derivative=sigp):
        self.weights = np.random.randn(layer_size, input_size)
        self.biases = np.random.randn(layer_size, 1)

        # # For tracking changes
        # self.orig_weights = self.weights.copy()
        # self.orig_biases = self.biases.copy()

        super().__init__(layer_size, input_size, activation_function, activation_derivative)

    def _propagate(self, input):
        # print(self.weights.shape)
        # print(input.shape)

        np.matmul(self.weights, input)
        layer_inputs = np.add(np.matmul(self.weights, input), self.biases)

        return layer_inputs

    def _backpropagate(self, layer_error, previous_inputs):
        # print(self.weights.transpose().shape)
        # print(layer_error.shape)

        return np.matmul(self.weights.transpose(), layer_error)

    # Currently test is unused
    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        # print(len(previous_activations))
        # print(len(self.past_errors))

        # TODO Change all previous to previous_layer

        assert len(previous_activations) == len(self.past_errors)
        minibatch_size = len(previous_activations)

        error_activations = [np.matmul(error, activation.transpose())
                             for error, activation in zip(self.past_errors, previous_activations)]

        self.weights = np.subtract(self.weights, (learning_rate / minibatch_size) * sum(error_activations))
        self.biases = np.subtract(self.biases, (learning_rate / minibatch_size) * sum(self.past_errors))


class OutputLayer(FullyConnectedLayer):
    def __init__(self, input_size, layer_size):
        super().__init__(input_size, layer_size, activation_function=sig, activation_derivative=None)

    def output_error(self, actual, expected):
        error = np.multiply(self.calc_cost_partial_derivs(expected, actual), sigp(self.last_inputs))
        return error

    def calc_cost(self, expected, final_activations):
        return np.square(np.sum(np.subtract(expected - final_activations))) * 0.5

    def calc_cost_partial_derivs(self, expected, final_activations):
        return np.subtract(final_activations, expected)


class Network:
    def __init__(self, layers=None):
        if len(layers) < 1:
            print("Error: There must be at least 1 layers.")
            raise ValueError

        self.layers = layers

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

    def gradient_descent(self, minibatch_inputs, minibatch_outputs, learning_rate):

        assert len(minibatch_inputs) == len(minibatch_outputs)

        lr_list = isinstance(learning_rate, list)

        if lr_list:
            assert len(learning_rate) == len(self.layers)
        else:
            learning_rate = [learning_rate]*len(self.layers)

        for inputs, output in zip(minibatch_inputs, minibatch_outputs):
            result = self.propagate(inputs, train=True)
            self.backpropagate(inputs, output, result)

        for l in range(len(self.layers) - 1, 0, -1):
            self.layers[l].gradient_decent(learning_rate[l], self.layers[l-1].past_activations)

        # TODO This has to be done after since it deletes cached activations, change caching?
        self.layers[0].gradient_decent(learning_rate[0], minibatch_inputs, True)

    def load(self, pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                self.layers = pickle.load(f)
        except FileNotFoundError:
            pass

    def train(self, epochs, minibatch_size, training_data, test_data, learning_rate=None, decay_rate=None, epoch_size_limit=None, validation_data=None, save_file=''):
        epoch_cnt = 0

        for epoch in range(epochs):
            if decay_rate is not None and not isinstance(learning_rate, list):
                epoch_lr = (1 / (1 + decay_rate*epoch_cnt))*learning_rate
            else:
                epoch_lr = learning_rate

            results = []
            results.append("Epoch " + str(epoch_cnt))
            results.append("Learning rate: " + str(epoch_lr))

            training_data_used = self.train_epoch(minibatch_size, training_data, epoch_lr, epoch_size_limit, validation_data)

            # Testing data results are ints while training data results are one-hot vectors
            results.append("Train performance: " + str(self.test([[example[0].reshape(1, 28, 28), np.argmax(example[1])] for example in training_data_used])))
            results.append("Test performance: " + str(self.test(test_data)))

            print('{0:<10} | {1:<40} | {2:<40} | {3:<40}'.format(*results))

            if save_file:
                with open(save_file, 'wb') as f:
                    pickle.dump(self.layers, f)

            epoch_cnt += 1

    def train_epoch(self, minibatch_size, training_data, learning_rate, epoch_size_limit, validation_data=None):
        random.shuffle(training_data)

        if epoch_size_limit is not None:
            training_data = training_data[:epoch_size_limit]

        minibatches = [training_data[k * minibatch_size:(k + 1) * minibatch_size] for k in
                       range(math.floor(len(training_data) / minibatch_size))]

        cnt = 0
        for minibatch in minibatches:
            minibatch_inputs = [example[0] for example in minibatch]
            # minibatch_inputs = [example[0] for example in minibatch]
            minibatch_outputs = [example[1] for example in minibatch]

            # start = time.time()
            self.gradient_descent(minibatch_inputs, minibatch_outputs, learning_rate)
            # end = time.time()
            # print('whole gradient decent: ')
            # print(end - start)

            cnt += minibatch_size
            # if cnt > 5000:
            #     return

            # TODO Make verbose option
            # print(str(cnt) + ' training examples exhausted.')

        return training_data

    def test(self, test_data):
        successful = 0
        for example in test_data:
            inputs = np.reshape(example[0], (28, 28))
            output = example[1]

            result = self.propagate(inputs)

            if np.argmax(result) == output:
                successful += 1

        return successful / len(test_data) * 100


class FlatteningLayer(Layer):
    def __init__(self, input_dimensions):
        assert isinstance(input_dimensions, tuple)

        self.input_dimensions = input_dimensions
        self.layer_size = 1;
        for dim in input_dimensions:
            self.layer_size *= dim

        # Currently Layer params are unused, initialize with no activation function.
        super().__init__(0, 0, activation_function=None, activation_derivative=None)

    def _propagate(self, inputs):
        return inputs.reshape((self.layer_size, 1))

    def _backpropagate(self, layer_error, previous_inputs):
        return layer_error.reshape(self.input_dimensions)

    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        pass


class ConvolutionalLayer(Layer):
    def __init__(self, input_dimensions, layer_size, convolution_size, stride, filter_count, activation_function=sig, activation_derivative=sigp):
        # For now only do 2d convolutions
        # assert len(input_dimensions) == 2
        # assert len(convolution_size) == 2

        if len(input_dimensions) == 2:
            input_dimensions = (1, 1) + input_dimensions
        elif len(input_dimensions) == 3:
            input_dimensions = (1,) + input_dimensions
        else:
            print('Invalid ConvolutionalLayer input dimensions.')
            raise ValueError

        # TODO layersize unused

        input_size = 1;

        for dim in input_dimensions:
            input_size *= dim

        super().__init__(input_size, layer_size, activation_function, activation_derivative)

        self.stride = stride

        self.feature_count = input_dimensions[1]

        self.conv_weights = np.random.randn(filter_count, self.feature_count, convolution_size, convolution_size)
        self.conv_bias = np.random.randn(filter_count, 1)

        self.conv_width = convolution_size
        self.conv_count = filter_count
        self.input_dims = input_dimensions

        # TODO Should be ceil?
        self.m_strides = int((self.input_dims[2] - self.conv_width) / self.stride + 1)
        self.n_strides = int((self.input_dims[3] - self.conv_width) / self.stride + 1)

        self.past_conv_errors = []

    def _propagate(self, inputs):
        # print(self.m_strides)
        # print(self.n_strides)

        # 0.016 sec
        # TODO Currently only supports 2d input (Not 3d)

        # activations = np.zeros((self.m_strides, self.n_strides))
        # layer_inputs = np.zeros((self.m_strides, self.n_strides))

        # for m in range(self.m_strides):
        #     for n in range(self.n_strides):
        #
        #
        #         input_slice = inputs[m*self.stride: m*self.stride + self.conv_width,
        #                              n*self.stride: n*self.stride + self.conv_width]
        #         layer_input = np.tensordot(input_slice, self.conv_weights, axes=((0, 1), (0, 1))) + self.conv_bias
        #         activation = sig(layer_input)
        #
        #         # TODO [,] or [][] for efficiency?
        #         activations[m, n] = activation
        #         layer_inputs[m, n] = layer_input

        #-------------------------------------------------------
        #
        # start = time.time()
        #
        # input_col = np.zeros((self.conv_width*self.conv_width, self.n_strides * self.m_strides))
        # w_row = self.conv_weights.reshape((1, self.conv_width*self.conv_width))
        #
        # # TODO Make into large matrix rep
        # for m in range(self.m_strides):
        #     for n in range(self.n_strides):
        #         input_slice = inputs[m*self.stride: m*self.stride + self.conv_width,
        #                              n*self.stride: n*self.stride + self.conv_width]
        #         col = input_slice.reshape((self.conv_width*self.conv_width, 1))
        #         #print(input_col[:,m*self.conv_width + n])
        #         input_col[:, m*self.conv_width + n] = col[:, 0]
        #
        # layer_inputs = np.dot(w_row, input_col).reshape((self.m_strides, self.n_strides))
        # activations = sig(layer_inputs)
        #
        # end = time.time()
        # # print('time to propagate: ')
        # # print(end - start)
        #
        #----------------------------------------------------------

        start = time.time()

        #print(inputs.reshape((1, 1) + inputs.shape).shape)

        input_col = im2col_indices(inputs.reshape(self.input_dims),
                                   self.conv_width,
                                   self.conv_width,
                                   padding=0,
                                   stride=self.stride)

        self.past_input_col = input_col

        w_row = self.conv_weights.reshape((self.conv_count, -1))
        # print(self.conv_weights.shape)
        layer_inputs = np.dot(w_row, input_col) + self.conv_bias
        # TODO Ensure all reshapes are preforming in correct order
        layer_inputs = layer_inputs.reshape(self.conv_count, self.m_strides, self.n_strides)

        end = time.time()
        # print('time to propagate: ')
        # print(end - start)

        return layer_inputs

    def _backpropagate(self, layer_error, previous_inputs):
        # 0.07 sec

        # TODO Currently only supports 2d input (Not 3d)

        # conv_error = np.zeros((self.conv_width, self.conv_width))
        # input_error = np.zeros(self.input_dims)
        #
        # for m in range(self.m_strides):
        #     for n in range(self.n_strides):
        #         input_slice = previous_inputs[m*self.stride: m*self.stride + self.conv_width,
        #                                       n*self.stride: n*self.stride + self.conv_width]
        #
        #
        #         conv_error = np.add(conv_error, input_slice*previous_inputs[m,n])
        #
        #         input_error_slice = input_error[m*self.stride: m*self.stride + self.conv_width,
        #                                         n*self.stride: n*self.stride + self.conv_width]
        #
        #         input_error[m * self.stride: m * self.stride + self.conv_width,
        #                     n * self.stride: n * self.stride + self.conv_width] = \
        #             np.add(input_error_slice, self.conv_weights*previous_inputs[m,n])

        #-------------------------------------------------------------------

        # conv_error_cols = np.zeros((self.conv_width*self.conv_width, self.n_strides*self.m_strides))
        #
        # # This many strides needed such that two convs dont add to the same input
        # non_interference_stride = math.ceil((self.conv_width - 1)/self.stride)
        #
        # # Stack of input errors to then collapse with a np.sum
        # non_interference_input_errors = np.zeros(self.input_dims + (non_interference_stride, non_interference_stride))
        #
        # start = time.time()
        #
        # for m in range(self.m_strides):
        #     for n in range(self.n_strides):
        #         input_slice = previous_inputs[m*self.stride: m*self.stride + self.conv_width,
        #                                       n*self.stride: n*self.stride + self.conv_width]
        #
        #         conv_error_cols[:,m*self.conv_width+n] = input_slice.reshape((self.conv_width*self.conv_width, 1))[:, 0]
        #
        # input_col = previous_inputs.reshape((1, previous_inputs.size))    #np.zeros((self.conv_width * self.conv_width, previous_inputs.size))
        # w_row = self.conv_weights.reshape(( self.conv_width * self.conv_width, 1))
        #
        # weighted_previous_inputs = np.dot(w_row, input_col).reshape(previous_inputs.shape + (self.conv_width, self.conv_width))
        #
        # #self.conv_weights * previous_inputs[m_location, n_location]
        #
        #
        #
        # #for
        #
        # for m_layer in range(non_interference_stride):
        #     for n_layer in range(non_interference_stride):
        #
        #         for m in range(math.ceil(self.m_strides/non_interference_stride)):
        #             if m*non_interference_stride >= self.m_strides:
        #                 continue
        #
        #             for n in range(math.ceil(self.n_strides / non_interference_stride)):
        #                 if n * non_interference_stride >= self.n_strides:
        #                     continue
        #
        #                 m_location = m*self.stride*non_interference_stride
        #                 n_location = n*self.stride*non_interference_stride
        #
        #                 # Assumes 2 input dims, change for 3d input (m,n,:)
        #                 non_interference_input_errors[
        #                     m_layer + m_location: m_layer + m_location + self.conv_width,
        #                     n_layer + n_location: n_layer + n_location + self.conv_width,
        #                     m_layer,
        #                     n_layer] = weighted_previous_inputs[m_location, n_location]
        #                 #self.conv_weights*previous_inputs[m_location, n_location]
        #
        # end = time.time()
        # # print('time to backpropagate: ')
        # # print(end - start)
        #
        # input_error = np.sum(non_interference_input_errors, axis=(2, 3))
        #
        # conv_error = np.sum(conv_error_cols, axis=1).reshape((self.conv_width, self.conv_width))
        #
        # self.past_conv_errors.append(conv_error)

        #---------------------------------------------------------------------

        # TODO Change caching to be defined by subclass

        start = time.time()

        input_col = self.past_input_col

        # layer_error = np.multiply(layer_error, relup(self.last_inputs))
        layer_error_row = layer_error.reshape((self.conv_count, self.m_strides * self.n_strides))

        conv_error = np.dot(layer_error_row, np.transpose(input_col)).reshape(self.conv_weights.shape)

        # TODO Filter rotation needed?
        w_row = self.conv_weights.reshape((self.conv_count, -1))
        # w_row = np.rot90(self.conv_weights, 2, (2, 3)).reshape((self.conv_count, -1))

        # Filter dimension is lost, all filter weight layer_errors are added together
        input_error_col = np.dot(np.transpose(w_row), layer_error_row)

        # print(input_error_col.shape)

        input_error = col2im_indices(input_error_col,
                                     self.input_dims,
                                     self.conv_width,
                                     self.conv_width,
                                     padding=0,
                                     stride=self.stride)

        self.past_conv_errors.append(conv_error)

        end = time.time()
        # print('time to backpropagate: ')
        # print(end - start)

        # input_error = np.multiply(input_error, sigp(previous_inputs))

        return input_error

    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        # print(len(previous_activations))
        # print(len(self.past_conv_errors))

        # TODO Figure out gradient decent
        # print(len(previous_activations))
        # print(len(self.past_conv_errors))
        assert len(previous_activations) == len(self.past_conv_errors)
        minibatch_size = len(previous_activations)

        filter_layer_errors = [np.sum(error, axis=(2, 3)).reshape(self.conv_count, 1) for error in self.past_errors]
        # TODO Lowering the filter size (increasing number of filter applications) requires decrease in learning rate?
        # TODO Make learning rate configurable per layer?
        self.conv_weights = \
            np.subtract(self.conv_weights, (learning_rate / minibatch_size) * sum(self.past_conv_errors))
        self.conv_bias = \
            np.subtract(self.conv_bias, (learning_rate / minibatch_size) * sum(filter_layer_errors))

        self.past_conv_errors = []


# TODO Currently unused, may not be stable since activation functions are also built into Layer base class.
class MaxPoolingLayer(Layer):
    def __init__(self, input_dimensions, layer_size, pool_width, stride):
        # if len(input_dimensions) == 2:
        #     input_dimensions = (1, 1) + input_dimensions
        # elif len(input_dimensions) == 3:
        #     input_dimensions = (1,) + input_dimensions
        if not len(input_dimensions) == 4:
            print('Invalid ConvolutionalLayer input dimensions.')
            raise ValueError

        input_size = 1
        for dim in input_dimensions:
            input_size *= dim

        super().__init__(input_size, layer_size, activation_function=None, activation_derivative=None)

        self.stride = stride
        self.pool_width = pool_width

        self.feature_count = input_dimensions[1]
        self.input_dims = input_dimensions

        # TODO Should be ceil?
        self.m_strides = int((self.input_dims[2] - self.pool_width) / self.stride + 1)
        self.n_strides = int((self.input_dims[3] - self.pool_width) / self.stride + 1)

    def _propagate(self, inputs):
        input_col = im2col_indices(inputs.reshape((self.input_dims[0]*self.input_dims[1], 1) + self.input_dims[2:]),
                                   self.pool_width,
                                   self.pool_width,
                                   padding=0,
                                   stride=self.stride)

        self.past_input_col = input_col
        self.past_input_indices = np.argmax(input_col, axis=0)

        layer_inputs = input_col[self.past_input_indices, range(self.past_input_indices.size)]
        layer_inputs = layer_inputs.reshape(self.m_strides, self.n_strides, 1, self.feature_count)
        layer_inputs = layer_inputs.transpose(2, 3, 0, 1)

        return layer_inputs

    def _backpropagate(self, layer_error, previous_inputs):
        input_col = self.past_input_col

        layer_error_row = layer_error.transpose(2, 3, 0, 1).reshape(-1) # layer_error.reshape((self.feature_count, self.m_strides * self.n_strides))

        input_error_col = np.zeros_like(input_col)
        input_error_col[self.past_input_indices, range(self.past_input_indices.size)] = layer_error_row

        input_error = col2im_indices(input_error_col,
                                     (self.input_dims[0]*self.input_dims[1], 1) + self.input_dims[2:],
                                     self.pool_width,
                                     self.pool_width,
                                     padding=0,
                                     stride=self.stride)

        input_error = input_error.reshape(self.input_dims)

        return input_error

    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        pass


# TODO Currently unused, may not be stable since activation functions are also built into Layer base class.
class ActivationLayer(Layer):
    def __init__(self, function, derivative):

        self.function = function
        self.derivative = derivative
        self.last_propagate_inputs = None

        # Currently Layer params are unused
        super().__init__(0, 0)

    def _propagate(self, inputs):
        self.last_propagate_inputs = inputs
        return self.function(inputs)

    def _backpropagate(self, layer_error, previous_inputs):
        return np.multiply(layer_error, self.derivative(self.last_propagate_inputs))

    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        pass


EPOCHS = 30
MINIBATCH_SIZE = 64


def main():
    np.random.seed(1234)
    random.seed(5678)

    n = Network([ConvolutionalLayer((1, 28, 28), 0,
                                    convolution_size=5,
                                    stride=1,
                                    filter_count=16),
                                    # activation_function=relu,
                                    # activation_derivative=relup),
                 MaxPoolingLayer((1, 16, 24, 24), 0, pool_width=2, stride=2),
                 # ActivationLayer(sig, sigp),

                 ConvolutionalLayer((16, 12, 12), 0,
                                    convolution_size=5,
                                    stride=1,
                                    filter_count=16),
                                    # activation_function=relu,
                                    # activation_derivative=relup),
                 MaxPoolingLayer((1, 16, 8, 8), 0, pool_width=2, stride=2),
                 # ActivationLayer(sig, sigp),

                # Pooling steps messing up?
                 FlatteningLayer((1, 16, 4, 4)),
                 FullyConnectedLayer(256, 100),
                 # ActivationLayer(sig, sigp),

                 #FullyConnectedLayer(30, 30),
                 OutputLayer(100, 10)])

                 # OutputLayer(3200, 10)])

    # n = Network([FlatteningLayer((28, 28)),
    #              FullyConnectedLayer(784, 30),
    #              FullyConnectedLayer(30, 30),
    #              OutputLayer(30, 10)])
    #
    # # 89.05999999999999
    # # 89.38000000000001
    # # 89.47
    # # 89.34
    # # 89.95
    # # 89.64

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', default='', help='Save network after every epoch to the provided filename.')
    parser.add_argument('-t', '--test', default='', help='Test saved network.')
    args = parser.parse_args()

    instance_filename = ''
    if args.save:
        instance_filename = args.save
        n.load(args.save)


    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # Reshape inputs
    training_data = [[example[0].reshape(1, 28, 28), example[1]] for example in training_data]
    test_data = [[example[0].reshape(1, 28, 28), example[1]] for example in test_data]
    # training_data = [[example[0].flatten(), example[1]] for example in training_data]
    # test_data = [[example[0].flatten(), example[1]] for example in test_data]

    if args.test:
        n.load(args.test)
        print(n.test(test_data))

    else:
        n.train(epochs=240,
                minibatch_size=64,
                training_data=training_data,
                learning_rate=0.3, #[0.006, None, 0.006, None, None, 3, 3],
                decay_rate=4*0.3/60,
                # epoch_size_limit=5000,
                test_data=test_data,
                save_file=instance_filename)


if __name__ == '__main__':
    main()
