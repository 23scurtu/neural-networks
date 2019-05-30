# This file provides the same network as in network.py, but using pytorch instead.

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
import torch.nn as nn
# torch.nn.functional as F
import torch
import torch.utils.data


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.second_layer = nn.Sequential(
            nn.Conv2d(16, 16, 5, stride=1),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256, 100),
            nn.Sigmoid(),
            nn.Linear(100, 10),
            nn.Sigmoid()
        )


    def forward(self, x):
        out = self.first_layer(x)
        out = self.second_layer(out)
        out = out.reshape(out.size(0), -1)
        # print(out.reshape(out.size(0), -1).shape)
        # exit()
        # print(out.shape)
        # exit()
        out = self.fc_layer(out)
        return out


def train(model, epochs, minibatch_size, training_data, test_data, learning_rate=None, decay_rate=None,
    epoch_size_limit=None, validation_data=None, save_file=''):
    epoch_cnt = 0

    loss_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # if decay_rate is not None and not isinstance(learning_rate, list):
        #     epoch_lr = (1 / (1 + decay_rate * epoch_cnt)) * learning_rate
        # else:
        #     epoch_lr = learning_rate

        results = []
        results.append("Epoch " + str(epoch_cnt))
        # results.append("Learning rate: " + str(epoch_lr))

        training_data_used = train_epoch(model, loss_criterion, optimizer, minibatch_size, training_data,
                                         learning_rate, epoch_size_limit, validation_data)

        model.eval()
        # Testing data results are ints while training data results are one-hot vectors
        results.append("Train performance: " + str(
            test(model, [[example[0].reshape(1, 28, 28), np.argmax(example[1])] for example in training_data_used])))
        results.append("Test performance: " + str(test(model, test_data)))

        model.train()

        print('{0:<10} | {1:<40} | {2:<40}'.format(*results))

        if save_file:
            with open(save_file, 'wb') as f:
                pickle.dump(model.layers, f)

        epoch_cnt += 1


def train_epoch(model, loss_criterion, optimizer, minibatch_size, training_data, learning_rate, epoch_size_limit, validation_data=None):
    random.shuffle(training_data)

    if epoch_size_limit is not None:
        training_data = training_data[:epoch_size_limit]

    inputs = [torch.Tensor(example[0]) for example in training_data]
    labels = [torch.Tensor(example[1]) for example in training_data]

    # print(np.stack(inputs, axis=0)[0])
    train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.stack(inputs, dim=0),
                                                                                      torch.stack(labels, dim=0)),
                                               batch_size=minibatch_size,
                                               shuffle=False)


    for inputs, labels in train_loader:
        # print(inputs[0])
        optimizer.zero_grad()

        outputs = model(inputs)
        # print(outputs.reshape(outputs.shape[0], outputs.shape[1], 1).shape)
        # print(labels.shape)
        # exit()
        loss = loss_criterion(outputs.reshape(outputs.shape[0], outputs.shape[1], 1), labels)


        loss.backward()
        optimizer.step()

    return training_data

    # minibatches = [training_data[k * minibatch_size:(k + 1) * minibatch_size] for k in
    #                range(math.floor(len(training_data) / minibatch_size))]
    #
    # cnt = 0
    # for minibatch in minibatches:
    #     minibatch_inputs = [example[0] for example in minibatch]
    #     # minibatch_inputs = [example[0] for example in minibatch]
    #     minibatch_outputs = [example[1] for example in minibatch]
    #
    #     # start = time.time()
    #     model.gradient_descent(minibatch_inputs, minibatch_outputs, learning_rate)
    #     # end = time.time()
    #     # print('whole gradient decent: ')
    #     # print(end - start)
    #
    #     cnt += minibatch_size
    #     # if cnt > 5000:
    #     #     return
    #
    #     # TODO Make verbose option
    #     # print(str(cnt) + ' training examples exhausted.')

    return training_data

def test(model, test_data):
    # inputs = [torch.Tensor(example[0]) for example in test_data]
    # labels = [torch.Tensor(example[1]) for example in test_data]
    #
    # test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.stack(inputs, dim=0),
    #                                                                                  torch.stack(labels, dim=0)),
    #                                           batch_size=1,
    #                                           shuffle=False)

    successful = 0
    for example in test_data:
        input = np.reshape(example[0], (1, 1, 28, 28))
        label = example[1]

        output = model(torch.Tensor(input))

        # print(output)

        if np.argmax(output.detach().numpy()) == label:
            successful += 1

    return successful / len(test_data) * 100



training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Reshape inputs
training_data = [[example[0].reshape(1, 28, 28), example[1]] for example in training_data]
test_data = [[example[0].reshape(1, 28, 28), example[1]] for example in test_data]
# training_data = [[example[0].flatten(), example[1]] for example in training_data]
# test_data = [[example[0].flatten(), example[1]] for example in test_data]

n = Network()
train(  model=n,
        epochs=240,
        minibatch_size=64,
        training_data=training_data,
        learning_rate=0.005, #[0.006, None, 0.006, None, None, 3, 3],
        decay_rate=4*0.3/60,
        epoch_size_limit=5000,
        test_data=test_data,
        save_file='')