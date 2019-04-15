import numpy as np
import math
import pprint
import mnist_loader
import random


def sig(x):
    # a = np.array([x])
    # np.exp(a)
    return 1/(1 + np.exp(-x))

#sig = np.vectorize(single_sig)


def sigp(x):
    return sig(x)*(1 - sig(x))

#sigp = np.vectorize(single_sigp)


class Network:
    def __init__(self, layers):
        if len(layers) < 2:
            print("Error: There must be at least 2 layers.")
            raise ValueError

        self.layers = layers
        self.depth = len(layers)

        self.weights = []
        self.biases = [np.random.randn(layersize, 1) for layersize in layers[1:]]
        self.layer_inputs = [np.zeros(layersize) for layersize in layers[1:]]
        self.layer_activations = [np.zeros(layersize) for layersize in layers]


        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]))


        self.learning_rate = 3

    def propogate(self, input):

        layer_activations = self.layer_activations.copy()
        layer_activations[0] = input

        layer_inputs = self.layer_inputs.copy()

        for i in range(self.depth - 1):
            layer_inputs[i] = np.add(np.matmul(self.weights[i], layer_activations[i]), self.biases[i])
            layer_activations[i+1] = sig(layer_inputs[i])

        return layer_activations, layer_inputs


    def calc_cost(self, expected, final_activations):
        return np.square(np.sum(np.subtract(expected - final_activations))) * 0.5


    def calc_cost_partial_derivs(self, expected, final_activations):
        #Note maybe switch
        return np.subtract(final_activations, expected)


    def backpropogate(self, expected, layer_inputs, layer_activations):
        output_error = np.multiply(self.calc_cost_partial_derivs(expected, layer_activations[-1]), sigp(layer_inputs[-1]))
        layer_errors = [output_error]

        for l in range(self.depth - 3, -1, -1):
            #TODO NOT COMING OUT AS VECTOR
            layer_error = np.multiply(np.matmul(self.weights[l+1].transpose(), layer_errors[0]), sigp(layer_inputs[l]))
            layer_errors = [layer_error] + layer_errors

        return layer_errors

    def gradient_descent(self, minibatch_inputs, minibatch_outputs):

        assert len(minibatch_inputs) == len(minibatch_outputs)

        error_activation_sums = [None]*self.depth
        error_bias_sums = [None]*self.depth

        for input, output in zip(minibatch_inputs, minibatch_outputs):
            layer_activations, layer_inputs = self.propogate(input)
            layer_errors = self.backpropogate(output, layer_inputs, layer_activations)

            for l in range(self.depth - 1):
                # layer_activations[l] is technically a^l-1
                if not (error_activation_sums[l] is None):
                    error_activation_sums[l] = np.add( error_activation_sums[l],
                                                       np.matmul(layer_errors[l], layer_activations[l].transpose()))
                else:
                    error_activation_sums[l] = np.matmul(layer_errors[l], layer_activations[l].transpose())

                if not (error_bias_sums[l] is None):
                    error_bias_sums[l] = np.add(error_bias_sums[l], layer_errors[l])
                else:
                    error_bias_sums[l] = layer_errors[l]

        for l in range(self.depth - 1):
            self.weights[l] = np.subtract(self.weights[l], (self.learning_rate/len(minibatch_inputs)) * error_activation_sums[l])
            self.biases[l] = np.subtract(self.biases[l], (self.learning_rate/len(minibatch_inputs)) * error_bias_sums[l])


def main():
    n = Network([784, 30, 10])
    # print(n.weights)
    # n.propogate(np.array([0.5,0.5,0.5]))
    # n.backpropogate(np.array([0.5,0.5,0.5]))

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # n = Network([3, 5, 5, 3])
    # minibatch_inputs = [np.array([0.5, 0.5, 0.5])]
    # minibatch_outputs = [np.array([0.5, 0.5, 0.5])]
    #
    # for i in range(1):
    #     n.gradient_descent(minibatch_inputs, minibatch_outputs)
    #     #pprint.pprint(n.weights)

    #pprint.pprint(training_data[0])



    EPOCHS = 30
    MINIBATCH_SIZE = 32


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
            input = example[0]
            output = example[1]

            activations, _ = n.propogate(input)
            result = activations[-1]

            if np.argmax(result) == output:
                successful += 1


        print(successful/TEST_COUNT*100)




if __name__ == '__main__':
    main()