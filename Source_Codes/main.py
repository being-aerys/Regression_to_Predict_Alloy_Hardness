import time, numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import copy





class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.L1 = nn.Linear(input_dim, input_dim * 2)
        self.L2 = nn.Linear(input_dim * 2, input_dim * 4)
        self.L3 = nn.Linear(input_dim * 4, input_dim * 2)
        self.L4 = nn.Linear(input_dim * 2, output_dim)

    def forward(self, input):

        input = F.leaky_relu(self.L1(input))
        input = F.leaky_relu(self.L2(input))
        input = F.leaky_relu(self.L3(input))
        input = self.L4(input)
        return input


if __name__ == "__main__":

    '''Check for cuda devices'''
    print("Current cuda device is ", torch.cuda.current_device())
    print("Total cuda-supporting devices count is ", torch.cuda.device_count())
    print("Current cuda device name is ", torch.cuda.get_device_name(torch.cuda.current_device()))

    '''Retrieve data'''
    features = np.genfromtxt("../Data/alloy_data.csv", delimiter=",")
    data = features[1:] #Removing the column names from the data

    #The shape is now (95, 16).

    data_old = copy.deepcopy(data)

    '''Divide into training and testing data after shuffling data'''
    np.random.shuffle(data)

    assert np.array_equal(data_old, data) == False #making sure that shuffling worked

    train_to_test = .8
    training_data = data[:]
    testing_data = data[int(.8 * 95):]

    '''Fix hyperparameters'''
    batch_size = 1 #Since very less data, we can use stochastic gradient descent
    max_epochs = 10

    '''Create model'''
    network = NeuralNetwork(15,1)

    '''If cuda device available, move the model to cuda device'''
    if torch.cuda.device_count() > 0:
        network.cuda()  # moving the model to gpu if available

    '''Choose an appropriate loss function'''
    criterion = nn.L1Loss()

    '''Choose an optimizer'''
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum= 0.9)

    '''Keep track of some variables for plots'''
    epoch_list_for_the_plot = []
    training_loss_list = []
    testing_loss_list = []

    '''Set the network to training mode'''
    network.train()

    '''Training regimen'''
    epoch_num = 0

    print("\nTraining starts!!!\n")

    while epoch_num < max_epochs:

        #shuffle before every epoch
        np.random.shuffle(training_data)

        sample_num = 0

        while sample_num < len(training_data):
            feature, value = training_data[sample_num][:15], training_data[sample_num][15]

            '''Convert into torch tensors'''
            feature, value = torch.tensor(feature), torch.tensor(value)

            '''Convert numpy variables into torch variables to make GPU computations'''
            feature, value = Variable(feature).cuda(), Variable(value).cuda()
            #this will track the gradients of the variables

            '''Pass the data through the network'''
            output = network(feature)

            '''Calculate the loss'''
            loss = criterion(output, value)



            '''Calculate the gradients of all the loss with respect to the parameters (and values) in the network'''
            loss.backward()

            '''Clear gradients of the optimizer at each iteration'''
            optimizer.zero_grad()

            '''Do backpropagation using the gradients calculated'''
            optimizer.step()

        '''Calculate the training and testing loss after each epoch'''
        total_loss = 0
        '''Turn the evaluation mode on for the network'''
        network.eval()

        #For training data
        for idx in len(training_data):
            feature, value = training_data[idx]
            feature, value = Variable(feature).cuda(), Variable(value).cuda()
            output = network(feature)
            loss = criterion(output, value)
            total_loss += loss.data

        training_loss_list.append(total_loss)
        total_loss = 0

        #For testing data
        for idx in len(testing_data):
            feature, value = testing_data[idx]
            feature, value = Variable(feature).cuda(), Variable(value).cuda()
            output = network(feature)
            loss = criterion(output, value)
            total_loss += loss.data

        testing_loss_list.append(total_loss)


        '''Convert back to training mode'''
        network.train()

        epoch_list_for_the_plot.append(epoch_num)

        '''Plot the results for each epoch'''
        plt.figure(1)  # ------------------------------------------------------------Mode = figure(1) for plt
        plt.plot(epoch_list_for_the_plot, training_loss_list, 'g')  # pass array or list
        plt.plot(epoch_list_for_the_plot, training_loss_list, 'r')
        plt.xlabel("Number of Epochs")
        plt.ylabel("Training Loss")
        # plt.legend(loc='upper left')
        plt.gca().legend(('Training Loss', 'Testing Loss'))
        plt.title("Number of Epochs VS L1 Loss")
        plt.grid()

        plt.figure(2)  # -----------------------------------------------------------Mode = figure(2) for plt
        plt.plot(epoch_list_for_the_plot, training_loss_list, "g")
        plt.plot(epoch_list_for_the_plot, testing_loss_list, "r")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Testing Loss")
        plt.gca().legend(("Training Loss", "Testing Loss"))
        plt.title("Number of Epochs vs L1 Loss")
        plt.grid()

    print('Finished Training')
    print('Saving model now...')

    '''Save model'''

    plt.show()

