#Created on 15 January, 2020
#This intellectual property belongs to Aashish Adhikari.
#Any usage by others hereafter is required to cite the author.

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
        self.dropout_1 = nn.Dropout(p=0.5)
        self.L3 = nn.Linear(input_dim * 4, input_dim * 8)
        self.L4 = nn.Linear(input_dim * 8, input_dim * 4)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.L5 = nn.Linear(input_dim * 4, input_dim * 2)
        self.L6 = nn.Linear(input_dim * 2, output_dim)

    def forward(self, input):
        input = F.sigmoid(self.L1(input))
        input = F.sigmoid(self.L2(input))
        input = self.dropout_1(input)
        input = F.sigmoid(self.L3(input))
        input = F.sigmoid(self.L4(input))
        input = self.dropout_2(input)
        input = F.sigmoid(self.L5(input))
        input = self.L6(input)
        return input


if __name__ == "__main__":

    #torch.cuda.manual_seed(42)

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

    train_to_test = .9
    training_data = data[:int(train_to_test * 95)]
    testing_data = data[int(train_to_test * 95):]

    '''Fix hyperparameters'''
    batch_size = 1 #Since very less data, we can use stochastic gradient descent
    max_epochs = 200

    '''Create model'''
    network = NeuralNetwork(15,1) # We only care about the transfer temperature for now

    '''If cuda device available, move the model to cuda device'''
    if torch.cuda.device_count() > 0:
        network.cuda()  # moving the model to gpu if available

    '''Convert all the model parameters to float'''
    network = network.float()

    '''Choose an appropriate loss function'''
    criterion = nn.L1Loss()

    '''Choose an optimizer'''
    #optimizer = torch.optim.sgd(network.parameters(), lr=0.05, momentum= 0.9)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.05)


    '''Create a learning rate scheduler'''
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    # step_size = 2, after every 2 epoch, new_lr = lr*gamma

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

        '''Change the learning rate at each epoch'''
        scheduler.step()

        #shuffle before every epoch
        np.random.shuffle(training_data)

        sample_num = 0


        while sample_num < len(training_data):

            optimizer.zero_grad()
            # Sets all gradients to zero,
            # Basically deletes all gradients from the Parameters, which were passed to the optimizer.

            '''Clear gradients of the optimizer at each iteration'''

            feature, value = training_data[sample_num][:15], training_data[sample_num][15]
            value = [value]

            '''Convert into torch tensors'''
            feature, value = torch.tensor(feature), torch.tensor(value)


            '''Convert numpy variables into torch variables to make GPU computations'''
            feature, value = Variable(feature).cuda(), Variable(value).cuda()
            #this will track the gradients of the variables

            '''Pass the data through the network'''
            output = network(feature.float())


            '''Calculate the loss'''
            loss = criterion(output, value)


            '''Calculate the gradients of all the loss with respect to the parameters (and values) in the network'''
            loss.backward()

            '''Make sure the gradients are not 0 or NAN in the layers'''
            #print(network.L1.weight.grad)
            #If yes, then its gradient explosion, reduce the learning rate
            #time.sleep(2)


            '''Do backpropagation using the gradients calculated'''
            optimizer.step()


            sample_num += 1

        '''Reset the scheduler for next epoch'''
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1)
        # step_size = 2, after every 2 epoch, new_lr = lr*gamma

        '''Calculate the training and testing loss after each epoch'''
        total_training_loss = 0
        total_testing_loss = 0
        '''Turn the evaluation mode on for the network'''
        network.eval()
        #For training data
        for idx in range(len(training_data)):

            feature, value = training_data[idx][:15], training_data[idx][15]
            value = [value]
            feature, value = torch.tensor(feature), torch.tensor(value)
            feature, value = Variable(feature).cuda(), Variable(value).cuda()
            output = network(feature.float())
            loss = criterion(output, value)
            #print(loss)
            #time.sleep(1)
            total_training_loss += loss.data
            #print(total_training_loss)
            #time.sleep(1)


        training_loss_list.append(total_training_loss)




        #For test data
        for idx in range(len(testing_data)):
            feature, value = testing_data[idx][:15], testing_data[idx][15]
            value = [value]
            feature, value = torch.tensor(feature), torch.tensor(value)
            feature, value = Variable(feature).cuda(), Variable(value).cuda()
            output = network(feature.float())
            loss = criterion(output, value)
            total_testing_loss += loss.data

        testing_loss_list.append(total_testing_loss)

        print("Epoch num: ", epoch_num, "Training Loss: ", total_training_loss.data, "Testing Loss: ", total_testing_loss.data)



        '''Save the model after each 10 epochs'''
        torch.save(network.state_dict(), '../Saved_Models/Melting_Point_Prediction_'+str(epoch_num)+'_'+str(total_testing_loss) +'.pth')

        total_training_loss = 0
        total_testing_loss = 0

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

        epoch_num += 1

    print('Finished Training')
    print('Saving model now...')

    '''Save model'''

    plt.show()

