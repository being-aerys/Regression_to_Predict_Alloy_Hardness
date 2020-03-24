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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
from math import sqrt

#melting point and hardness


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.L1 = nn.Linear(input_dim, input_dim * 2)



        self.L2 = nn.Linear(input_dim * 2, input_dim * 4)

        #self.batchnorm_1 = nn.BatchNorm1d(input_dim * 4, 1e-12, affine=True, track_running_stats=True)



        self.dropout_1 = nn.Dropout(p=0.5)
        self.L3 = nn.Linear(input_dim * 4, input_dim * 8)



        self.L4 = nn.Linear(input_dim * 8, input_dim * 4)

        #self.batchnorm_2 = nn.BatchNorm1d(input_dim * 4, 1e-12, affine=True, track_running_stats=True)




        self.dropout_2 = nn.Dropout(p=0.5)
        self.L5 = nn.Linear(input_dim * 4, input_dim * 2)
        self.L6 = nn.Linear(input_dim * 2, output_dim)

    def forward(self, input):
        input = F.leaky_relu(self.L1(input))

        input = self.L2(input)

        # input = input.unsqueeze(0) #https://discuss.pytorch.org/t/batchnorm1d-valueerror-expected-2d-or-3d-input-got-1d-input/42081
        #
        # input = self.batchnorm_1(input)

        input = F.tanh(input)

        input = self.dropout_1(input)

        input = F.leaky_relu(self.L3(input))

        input = self.L4(input)

        # input = input.unsqueeze(0)
        #
        # input = self.batchnorm_2(input)

        input = F.tanh(input)

        input = self.dropout_2(input)

        input = F.leaky_relu(self.L5(input))

        input = self.L6(input)

        return input


if __name__ == "__main__":

    #torch.cuda.manual_seed(42)

    '''Check for cuda devices'''
    print("Current cuda device is ", torch.cuda.current_device())
    print("Total cuda-supporting devices count is ", torch.cuda.device_count())
    print("Current cuda device name is ", torch.cuda.get_device_name(torch.cuda.current_device()))

    '''Retrieve data'''
    features_with_label_with_column_names = np.genfromtxt("../Data/newdata_csv.csv", delimiter=",")
    print("\nPrinting raw data now!!!\n")
    print(features_with_label_with_column_names)

    print("The shape is: ", features_with_label_with_column_names.shape)
    time.sleep(2)

    feature_with_labels = features_with_label_with_column_names[1:] #Removing the column names from the data

    print("Removing the column names from the raw data.\n")
    print(feature_with_labels)
    time.sleep(2)

    old_feature_with_labels_only = copy.deepcopy(feature_with_labels)

    '''Divide into training and testing data after shuffling data'''
    print("\nShuffling the data.\n")
    np.random.shuffle(feature_with_labels)

    assert np.array_equal(old_feature_with_labels_only, feature_with_labels) == False #making sure that shuffling worked

    '''Normlaize the data including the target'''
    scaler = StandardScaler()
    scaler.fit(feature_with_labels)

    print("Then mean values for all the features are: ",scaler.mean_)
    print("\nThe variances for all the features are: ", scaler.var_)

    feature_with_labels = scaler.transform(feature_with_labels)




    '''DO PCA to remove redundant features, but make sure you remove the target column first.'''
    last_col = feature_with_labels.shape[1] - 1
    data_without_target_column = copy.deepcopy(feature_with_labels[:,:last_col])
    target_column = copy.deepcopy(feature_with_labels[:,last_col])
    pca = PCA(0.95)

    print("1",data_without_target_column[0])
    pca.fit(data_without_target_column)
    print("The number of features that contain 95% variance in the data set is: ",pca.n_components_ )

    print("The shape of features matrix before dimension reduction is: ", data_without_target_column.shape)

    '''Use only those components'''
    transformed_data_without_target_column = pca.transform(data_without_target_column)

    print("The shape of features matrix after dimension reduction is: ", transformed_data_without_target_column.shape)


    #shuffling garnu aghi combine the features and the target
    '''Combine the features and the labels'''
    reshaped_target_column = target_column.reshape((203,1))

    data = np.concatenate((transformed_data_without_target_column,reshaped_target_column),axis=1)

    print("After concatenation, the shape of the data is: ", data.shape)
    print("Dividing into training and testing data.")

    '''Train-Test split'''
    train_to_test = .9
    training_data = data[:int(train_to_test * data.shape[0])]
    testing_data = data[int(train_to_test * data.shape[0]):]


    '''Fix hyperparameters'''
    batch_size = 1 #Since very less data, we can use stochastic gradient descent
    max_epochs = 1000

    '''Create model'''
    network = NeuralNetwork(13,1) # We only care about the melting temperature for now

    '''If cuda device available, move the model to cuda device'''
    if torch.cuda.device_count() > 0:
        network.cuda()  # moving the model to gpu if available

    '''Convert all the model parameters to float'''
    network = network.float()

    '''Choose an appropriate loss function'''
    criterion = nn.MSELoss()

    '''Choose an optimizer'''
    #optimizer = torch.optim.sgd(network.parameters(), lr=0.05, momentum= 0.9)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=0.05, )


    '''Create a learning rate scheduler'''
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    # step_size = 2, after every 2 epoch, new_lr = lr*gamma

    '''Keep track of some variables for plots'''
    epoch_list_for_the_plot = []
    average_training_loss_list = []
    average_testing_loss_list = []

    '''Training regimen'''
    epoch_num = 0

    print("\nTraining starts!!!\n")

    testing_loss_comparator = sys.maxsize

    while epoch_num < max_epochs:

        '''Change the learning rate at each epoch'''
        #scheduler.step()

        #shuffle before every epoch
        np.random.shuffle(training_data)

        sample_num = 0

        '''Set the network to training mode'''
        network.train()

        '''Find the target column num'''
        shape_of_data = training_data.shape
        target_column_num = shape_of_data[1] - 1


        while sample_num < len(training_data):

            optimizer.zero_grad()
            # Sets all gradients to zero,
            # Basically deletes all gradients from the Parameters, which were passed to the optimizer.

            '''Clear gradients of the optimizer at each iteration'''

            feature, value = training_data[sample_num][:target_column_num], training_data[sample_num][target_column_num]
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


            '''Do gradient clipping to avoid spikes in the training and testing loss'''
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1)

            '''Do backpropagation using the gradients calculated'''
            optimizer.step()


            sample_num += 1

        '''Reset the scheduler for next epoch'''
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1)
        # step_size = 2, after every 2 epoch, new_lr = lr*gamma

        '''Calculate the training and testing loss after each epoch'''
        total_training_loss = 0
        total_testing_loss = 0
        '''Turn the evaluation mode on for the network'''
        network.eval()
        #For training data
        for idx in range(len(training_data)):

            feature, value = training_data[idx][:target_column_num], training_data[idx][target_column_num]
            value = [value]
            feature, value = torch.tensor(feature), torch.tensor(value)
            feature, value = Variable(feature).cuda(), Variable(value).cuda()
            output = network(feature.float())

            loss = criterion(output, value)

            total_training_loss += loss.data



        average_training_loss_list.append(total_training_loss/len(training_data))


        total_testing_loss = 0

        #For test data
        for idx in range(len(testing_data)):
            feature, value = testing_data[idx][:target_column_num], testing_data[idx][target_column_num]
            value = [value]
            feature, value = torch.tensor(feature), torch.tensor(value)
            feature, value = Variable(feature).cuda(), Variable(value).cuda()
            output = network(feature.float())
            loss = criterion(output, value)
            total_testing_loss += loss.data

        average_testing_loss_list.append(total_testing_loss/len(testing_data))

        # print("Epoch num: ", epoch_num, "Average Training Loss: ", total_training_loss/len(training_data), "Average Testing Loss: ", total_testing_loss/len(testing_data))



        '''Save the model if the testing loss decreases'''
        if total_testing_loss/len(testing_data) < testing_loss_comparator:
            print("Epoch num: ", epoch_num, "Average Training Loss: ", total_training_loss / len(training_data),
                  "Average Testing Loss: ", total_testing_loss / len(testing_data))

            print("Found a model with a decreased testing loss in epoch ", epoch_num, ". Saving the model.\n")
            torch.save(network.state_dict(), '../Saved_Models/Melting_Point_'+str(epoch_num)+'.pth')
            testing_loss_comparator = total_testing_loss/len(testing_data)
        else:
            print("Epoch num: ", epoch_num)

        total_training_loss = 0
        total_testing_loss = 0

        '''Convert back to training mode'''
        network.train()

        epoch_list_for_the_plot.append(epoch_num)

        '''Plot the results for each epoch'''
        plt.figure(1)  # ------------------------------------------------------------Mode = figure(1) for plt
        plt.plot(epoch_list_for_the_plot, average_training_loss_list, 'g')  # pass array or list
        #plt.plot(epoch_list_for_the_plot, training_loss_list, 'r')
        plt.xlabel("Number of Epochs")
        plt.ylabel("Training Loss")
        # plt.legend(loc='upper left')
        plt.gca().legend(('Training Loss',))
        plt.title("Number of Epochs VS L1 Loss")
        plt.grid()

        plt.figure(2)  # -----------------------------------------------------------Mode = figure(2) for plt
        #plt.plot(epoch_list_for_the_plot, training_loss_list, "g")
        plt.plot(epoch_list_for_the_plot, average_testing_loss_list, "r")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Testing Loss")
        plt.gca().legend(("Testing Loss",))
        plt.title("Number of Epochs vs L1 Loss")
        plt.grid()

        epoch_num += 1

    print('Finished Training')


    '''Save model'''

    plt.show()


    #shuffle all dataset except the test sample, standardize using the values of the test sample as well
    # divide into train and test then do testing of the test sample along other test samples and print the last of the test sample i.e.,
    #alloy of interest

    #412

