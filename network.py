import numpy as np
import math
import pprint
import mnist_loader
import random
import sys
import timeit
import time
from im2col import *

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

    def _backpropagate(self, layer_error, previous_inputs):
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


class FlatteningLayer(Layer):
    def __init__(self, input_dimensions):
        assert isinstance(input_dimensions, tuple)

        self.input_dimensions = input_dimensions
        self.layer_size = 1;
        for dim in input_dimensions:
            self.layer_size *= dim

        # Currently Layer params are unused
        super().__init__(0, 0)

    def _propagate(self, inputs):
        return inputs.reshape((self.layer_size,1)), inputs.reshape((self.layer_size,1))

    def _backpropagate(self, layer_error, previous_inputs):
        return layer_error.reshape(self.input_dimensions)

    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        pass


class ConvolutionalLayer(Layer):
    def __init__(self, input_dimensions, layer_size, convolution_size, stride, filter_count):
        # For now only do 2d convolutions
        assert len(input_dimensions) == 2
        # assert len(convolution_size) == 2

        # TODO layersize unused

        input_size = 1;

        for dim in input_dimensions:
            input_size *= dim

        super().__init__(input_size, layer_size)

        self.stride = stride

        self.conv_weights = np.random.randn(filter_count, convolution_size, convolution_size)
        self.conv_bias = np.random.randn(filter_count, 1)

        self.conv_width = convolution_size
        self.conv_count = filter_count
        self.input_dims = input_dimensions

        # TODO Should be ceil?
        self.m_strides = math.ceil((self.input_dims[0] - self.conv_width) / self.stride)
        self.n_strides = math.ceil((self.input_dims[1] - self.conv_width) / self.stride)

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

        input_col = im2col_indices(inputs.reshape((1, 1) + inputs.shape),
                                   self.conv_width,
                                   self.conv_width,
                                   padding=0,
                                   stride=self.stride)

        self.past_input_col = input_col

        w_row = self.conv_weights.reshape((self.conv_count, self.conv_width * self.conv_width))

        layer_inputs = np.dot(w_row, input_col) + self.conv_bias
        # TODO Ensure all reshapes are preforming in correct order
        layer_inputs = layer_inputs.reshape(self.conv_count, self.m_strides, self.n_strides)

        activations = sig(layer_inputs)

        end = time.time()
        # print('time to propagate: ')
        # print(end - start)

        return layer_inputs, activations

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

        layer_error_row = layer_error.reshape((self.conv_count, self.m_strides * self.n_strides))

        conv_error = np.dot(layer_error_row, np.transpose(input_col)).reshape(self.conv_count,
                                                                              self.conv_width,
                                                                              self.conv_width)

        w_row = self.conv_weights.reshape((self.conv_count, -1))

        # Filter dimension is lost, all filter weight layer_errors are added together
        input_error_col = np.dot(np.transpose(w_row), layer_error_row)

        # print(input_error_col.shape)

        input_error = col2im_indices(input_error_col,
                                     (1, 1) + self.input_dims,
                                     self.conv_width,
                                     self.conv_width,
                                     padding=0,
                                     stride=self.stride)

        self.past_conv_errors.append(conv_error)

        end = time.time()
        # print('time to backpropagate: ')
        # print(end - start)

        return input_error

    def _gradient_decent(self, learning_rate, previous_activations, test=False):
        # print(len(previous_activations))
        # print(len(self.past_conv_errors))

        # TODO Figure out gradient decent
        # print(len(previous_activations))
        # print(len(self.past_conv_errors))
        assert len(previous_activations) == len(self.past_conv_errors)
        minibatch_size = len(previous_activations)

        filter_layer_errors = [np.sum(error, axis=(1, 2)).reshape(self.conv_count, 1) for error in self.past_errors]
        self.conv_weights = \
            np.subtract(self.conv_weights, (learning_rate / minibatch_size) * sum(self.past_conv_errors))
        self.conv_bias = \
            np.subtract(self.conv_bias, (learning_rate / minibatch_size) * sum(filter_layer_errors))

        self.past_conv_errors = []

EPOCHS = 30
MINIBATCH_SIZE = 32


# TODO Move SGD code inside of Network
def main():
    n = Network([ConvolutionalLayer((28,28), 0, convolution_size=5,
                                    stride=1,
                                    filter_count=3),
                 FlatteningLayer((3, 23, 23)),
                 FullyConnectedLayer(1587, 30),
                 #FullyConnectedLayer(30, 30),
                 OutputLayer(30,10)])

    # n = Network([FullyConnectedLayer([28, 28], [5, 5]),
    #              #FullyConnectedLayer(30, 30),
    #              OutputLayer([5, 5], 10)])

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    for epoch in range(EPOCHS):
        random.shuffle(training_data)

        minibatches = [training_data[k*MINIBATCH_SIZE:(k+1)*MINIBATCH_SIZE] for k in range(math.floor(len(training_data)/MINIBATCH_SIZE))]

        cnt = 0
        for minibatch in minibatches:

            minibatch_inputs = [np.reshape(example[0], (28, 28)) for example in minibatch]
            # minibatch_inputs = [example[0] for example in minibatch]
            minibatch_outputs = [example[1] for example in minibatch]


            # start = time.time()
            n.gradient_descent(minibatch_inputs, minibatch_outputs)
            # end = time.time()
            # print('whole gradient decent: ')
            # print(end - start)

            cnt += MINIBATCH_SIZE
            # if cnt > 10000:
            #     break
            print(str(cnt) + ' training examples exhausted.')

        TEST_COUNT = len(test_data)
        successful = 0
        for example in test_data:
            inputs = np.reshape(example[0], (28, 28))
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
