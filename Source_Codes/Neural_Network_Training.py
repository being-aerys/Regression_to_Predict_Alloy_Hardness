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
import sys, os
from math import sqrt

#melting point and hardness

'''If cuda device present, use cuda, else use cpu.'''
#use_cuda = True if torch.cuda.device_count() > 0 else False
use_cuda = False #But in this case always use cpu



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
        #print(input)

        input = F.leaky_relu(self.L1(input))


        input = self.L2(input)

        # input = input.unsqueeze(0) #https://discuss.pytorch.org/t/batchnorm1d-valueerror-expected-2d-or-3d-input-got-1d-input/42081
        #
        # input = self.batchnorm_1(input)

        input = F.leaky_relu(input)

        input = self.dropout_1(input)

        input = F.leaky_relu(self.L3(input))

        input = self.L4(input)

        # input = input.unsqueeze(0)
        #
        # input = self.batchnorm_2(input)

        input = F.leaky_relu(input)

        input = self.dropout_2(input)

        input = F.leaky_relu(self.L5(input))

        input = self.L6(input)

        return input


if __name__ == "__main__":

    '''Check for cuda devices'''
    if use_cuda == True:
        #Set manual seed.
        torch.cuda.manual_seed(100)
        print("Current cuda device is ", torch.cuda.current_device())
        print("Total cuda-supporting devices count is ", torch.cuda.device_count())
        print("Current cuda device name is ", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("No cuda device available.")
    #torch.seed(100)



    '''Retrieve data'''
    #Use only 5 relevant features without composition information
    #features_with_label_with_column_names = np.genfromtxt("../Data/newdata_hardness_as_target_csv_limited_features.csv", delimiter=",")
    #Use all features including composition
    features_with_label_with_column_names = np.genfromtxt("../Data/newdata_hardness_as_target_csv_only_simulation_data.csv", delimiter=",")


    print("\nPrinting raw data now!!!\n")
    print(features_with_label_with_column_names)

    print("The shape is: ", features_with_label_with_column_names.shape, " and the data type is: ",features_with_label_with_column_names.dtype)
    #time.sleep(2)

    print("Removing the column names from the raw data.\n")
    feature_with_labels = features_with_label_with_column_names[1:] #Removing the column names from the data

    # print(feature_with_labels)
    # time.sleep(2)

    independent_copy_of_feature_with_labels_only = copy.deepcopy(feature_with_labels)

    '''Divide into training and testing data after shuffling data'''
    print("\nShuffling the data.\n")
    np.random.seed(100)
    np.random.shuffle(feature_with_labels)

    assert np.array_equal(independent_copy_of_feature_with_labels_only, feature_with_labels) == False #making sure that shuffling worked






    # '''DO PCA to remove redundant features, but make sure you remove the target column first.'''
    # last_col = feature_with_labels.shape[1] - 1
    # data_without_target_column = copy.deepcopy(feature_with_labels[:,:last_col])
    # target_column = copy.deepcopy(feature_with_labels[:,last_col])
    # pca = PCA(0.95)
    #
    # print("1",data_without_target_column[0])
    # pca.fit(data_without_target_column)
    # print("The number of features that contain 95% variance in the data set is: ",pca.n_components_ )
    #
    # print("The shape of features matrix before dimension reduction is: ", data_without_target_column.shape)
    #
    # '''Use only those components'''
    # transformed_data_without_target_column = pca.transform(data_without_target_column)
    #
    # print("The shape of features matrix after dimension reduction is: ", transformed_data_without_target_column.shape)
    #
    #
    # #shuffling garnu aghi combine the features and the target
    # '''Combine the features and the labels'''
    # reshaped_target_column = target_column.reshape((203,1))
    #
    # data = np.concatenate((transformed_data_without_target_column,reshaped_target_column),axis=1)
    #
    # print("After concatenation, the shape of the data is: ", data.shape)

    print("Dividing into training and testing data.")

    '''Train-Test split'''
    train_to_test = .9
    training_data = feature_with_labels[:int(train_to_test * feature_with_labels.shape[0])]
    testing_data = feature_with_labels[int(train_to_test * feature_with_labels.shape[0]):]

    # Calculate the num of input features
    num_features = feature_with_labels.shape[1] - 1


    print(num_features)

    input()

    '''Lets comment the code that appends a specific test sample to the testing set for now.'''
    #Append the test sample the we need to check explicitly at the end of the test set
    # print("Before adding the explicit test sample, the shape of the testing set is: ",testing_data.shape)
    # if num_features == 23:
    #     explixit_test_sample = np.array([[0,0,15,20,30,0,15,0,20,0,0,0,0,0,0,0,0,13.08, 246, 106, 5.65, 2956, 979]])
    #
    # else:
    #     explixit_test_sample = np.array([[13.08, 246, 106, 5.65, 2956, 979]])
    #
    # #Append a sample at the end of the numpy array, notice the double [] around the explicit test sample to make equal dimensions
    # testing_data = np.concatenate((testing_data,explixit_test_sample), axis=0)
    # print("After adding the explicit test sample, the shape of the testing set is: ",testing_data.shape)


    '''Normlaize both training and test data including the target using the parameters (mean, std, min, max ) generated from the training data only'''
    scaler_obj_from_training_data = StandardScaler()
    scaler_obj_from_training_data.fit(training_data)

    print("Then mean values for all the features are: ", scaler_obj_from_training_data.mean_)
    print("\nThe variances for all the features are: ", scaler_obj_from_training_data.var_)

    mean_val_hardness = scaler_obj_from_training_data.mean_[-1]
    variance_hardness = scaler_obj_from_training_data.var_[-1]

    scaled_training_data = scaler_obj_from_training_data.transform(training_data)
    scaled_testing_data = scaler_obj_from_training_data.transform(testing_data)



    '''Fix hyperparameters'''
    batch_size = 1 #Since very less data, we can use one sample for each update
    max_epochs = 200


    '''Create model'''
    network = NeuralNetwork(num_features,1) # We only care about the hardness temperature for now

    '''If cuda device available, move the model to cuda device'''
    if use_cuda == True:
        network.cuda()  # moving the model to gpu if available

    '''Convert all the model parameters to float'''
    network = network.float()

    '''Choose an appropriate loss function'''
    criterion = nn.L1Loss()

    '''Choose an optimizer'''
    #optimizer = torch.optim.sgd(network.parameters(), lr=0.05, momentum= 0.9)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002, )


    '''Create a learning rate scheduler'''
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1) #means multiply the learning rate by gamma after each 30 iterations
    # step_size = 2 means after every 2 epoch, new_lr = lr*gamma

    '''Keep track of some variables for plots'''
    epoch_list_for_the_plot = []
    average_training_loss_list = []
    average_testing_loss_list = []

    '''Training regimen'''
    epoch_num = 0

    print("\nTraining starts!!!\n")

    testing_loss_comparator = sys.maxsize

    while epoch_num < max_epochs:


        #print("\nLearning rate is: ", scheduler.get_lr())

        #shuffle before every epoch
        np.random.shuffle(scaled_training_data)
        #print("First sample in this epoch after shuffling is: ", training_data[0]," ,This should be different from the "
        #                                                                         "first sample in another epoch.")

        sample_num = 0

        '''Set the network to training mode.'''
        network.train()

        '''Find the target column num'''
        shape_of_data = scaled_training_data.shape
        target_column_num = shape_of_data[1] - 1


        while sample_num < len(scaled_training_data):

            optimizer.zero_grad()
            # Sets all gradients to zero,
            # Basically deletes all gradients from the Parameters, which were passed to the optimizer.

            '''Clear gradients of the optimizer at each iteration'''

            feature, value = scaled_training_data[sample_num][:target_column_num], scaled_training_data[sample_num][target_column_num]
            value = [value]

            '''If GPU available, convert the input to cuda FloatTensors, else torch FloatTensors.'''
            if use_cuda == True:
                feature, value = torch.cuda.FloatTensor(feature), torch.cuda.FloatTensor(value)
            else:
                feature, value = torch.FloatTensor(feature), torch.FloatTensor(value)

            '''Pass the data through the network.'''
            output = network(feature)


            '''Calculate the loss'''
            loss = criterion(output, value)


            '''Calculate the gradients of all the loss with respect to the parameters (and values) in the network'''
            loss.backward()

            '''Make sure the gradients are not 0 or NAN in the layers'''
            #print(network.L1.weight.grad)
            #If yes, then its gradient explosion, reduce the learning rate
            #time.sleep(2)


            #'''Do gradient clipping to avoid spikes in the training and testing loss'''
            #torch.nn.utils.clip_grad_norm_(network.parameters(), 1)

            '''Do backpropagation using the gradients calculated'''
            optimizer.step()


            sample_num += 1

        '''Calculate the training and testing loss after each epoch'''
        total_training_loss = 0
        total_testing_loss = 0
        '''Turn the evaluation mode on for the network'''
        network.eval()
        #For training data
        with torch.no_grad():
                for idx in range(len(scaled_training_data)):

                    feature, value = scaled_training_data[idx][:target_column_num], scaled_training_data[idx][target_column_num]
                    value = [value]
                    if use_cuda == True:
                        feature, value = torch.cuda.FloatTensor(feature), torch.cuda.FloatTensor(value)
                    else:
                        feature, value = torch.FloatTensor(feature), torch.FloatTensor(value)

                    output = network(feature)

                    loss = criterion(output, value)

                    total_training_loss += loss.data



        average_training_loss_list.append(total_training_loss/len(scaled_training_data))


        total_testing_loss = 0

        #For test data
        with torch.no_grad():
            for idx in range(len(testing_data)):
                feature, value = scaled_testing_data[idx][:target_column_num], scaled_testing_data[idx][target_column_num]
                value = [value]
                if use_cuda == True:
                    feature, value = torch.cuda.FloatTensor(feature), torch.cuda.FloatTensor(value)
                else:
                    feature, value = torch.FloatTensor(feature), torch.FloatTensor(value)

                output = network(feature)
                loss = criterion(output, value)
                total_testing_loss += loss.data

        average_testing_loss_list.append(total_testing_loss/len(scaled_testing_data))

        # print("Epoch num: ", epoch_num, "Average Training Loss: ", total_training_loss/len(training_data), "Average Testing Loss: ", total_testing_loss/len(testing_data))



        '''Save the model if the testing loss decreases'''
        print("Epoch num: ", epoch_num, "Average Training Loss: ", total_training_loss / len(scaled_training_data),
                  "Average Testing Loss: ", total_testing_loss / len(scaled_testing_data))
        if total_testing_loss / len(testing_data) < testing_loss_comparator:
            print("\nFound a model with a decreased testing loss in epoch ", epoch_num, ". Saving the model as: target_Hardness_features_"+str(num_features)+"_epochs_"+str(epoch_num)+'.pth\n')
            #Create directory if doesnt exist
            if not os.path.exists("Saved_Models_All_Features_only_simulation_data"):
                os.makedirs("Saved_Models_All_Features_only_simulation_data")

            torch.save(network.state_dict(), 'Saved_Models_All_Features_only_simulation_data/target_Hardness_features_'+str(num_features)+"_epochs_"+str(epoch_num)+'.pth')
            testing_loss_comparator = total_testing_loss/len(scaled_testing_data)


            '''Get the prediction for the test sample.'''
            test_network = NeuralNetwork(num_features,1)
            if num_features == 22:
                first_test_sample = np.array([[0, 0, 15, 20, 30, 0, 15, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 13.08, 246, 106, 5.65, 2956, 979]])
                second_test_sample = np.array([[3, 0, 11.9,20,30,0,15,0,20,0,0,0,0,0,0,0,0.1, 13.76,244.11,106.12,5.64,2935.7,988.9]])
            else:
                first_test_sample = np.array([[13.08, 246, 106, 5.65, 2956, 979]])
                second_test_sample = np.array([[13.76,244.11,106.12,5.64,2935.7,988.9]])

            scaled_first_test_sample = scaler_obj_from_training_data.transform(first_test_sample)
            last_feature = (scaled_first_test_sample.shape[1])-1
            features_test_sample_first = scaled_first_test_sample[0][:last_feature]
            tensor_form_first = torch.FloatTensor(features_test_sample_first)
            prediction_of_hardness_first = test_network(tensor_form_first)
            predicted_first = mean_val_hardness + (sqrt(variance_hardness) * prediction_of_hardness_first)
            print("Predicted hardness of the first test sample is: ", predicted_first,".\n")

            scaled_second_test_sample = scaler_obj_from_training_data.transform(second_test_sample)
            last_feature = (scaled_second_test_sample.shape[1]) - 1
            features_test_sample_second = scaled_second_test_sample[0][:last_feature]
            tensor_form_second = torch.FloatTensor(features_test_sample_second)
            prediction_of_hardness_second = test_network(tensor_form_second)
            predicted_second = mean_val_hardness + (sqrt(variance_hardness) * prediction_of_hardness_second)
            print("Predicted hardness of the second test sample is: ", predicted_second, ".\n")

        total_training_loss = 0
        total_testing_loss = 0

        '''Convert back to training mode'''
        network.train()

        epoch_list_for_the_plot.append(epoch_num)



        epoch_num += 1
        '''VVI: This step() method should be the last thing you call in the loop.'''
        #https://github.com/pytorch/pytorch/issues/22107
        scheduler.step()

    '''Plot the results for each epoch'''
    #plt.figure(1)  # ------------------------------------------------------------Mode = figure(1) for plt
    # plt.plot(epoch_list_for_the_plot, average_training_loss_list, 'g')  # pass array or list
    # # plt.plot(epoch_list_for_the_plot, training_loss_list, 'r')
    # plt.xlabel("Number of Epochs")
    # plt.ylabel("Training Loss")
    # # plt.legend(loc='upper left')
    # plt.gca().legend(('Training Loss',))
    # plt.title("Number of Epochs VS L1 Loss")
    # plt.grid()

    #plt.figure(2)  # -----------------------------------------------------------Mode = figure(2) for plt
    # plt.plot(epoch_list_for_the_plot, training_loss_list, "g")
    # plt.plot(epoch_list_for_the_plot, average_testing_loss_list, "r")
    # plt.xlabel("Number of Epochs")
    # plt.ylabel("Testing Loss")
    # plt.gca().legend(("Testing Loss",))
    # plt.title("Number of Epochs vs L1 Loss")
    # plt.grid()



    np.savetxt(fname='Saved_Models_All_Features_only_simulation_data/training_loss_list.csv', X=average_training_loss_list, delimiter=",")
    np.savetxt(fname='Saved_Models_All_Features_only_simulation_data/testing_loss_list.csv', X=average_testing_loss_list, delimiter=",")

    print('Finished Training')


    '''Plot results.'''

    epoch_list_for_the_plot = [i for i in range(1, max_epochs+1, 1)]

    training_loss = np.genfromtxt("Saved_Models_All_Features_only_simulation_data/training_loss_list.csv", delimiter=",")
    testing_loss_list = np.genfromtxt("Saved_Models_All_Features_only_simulation_data/testing_loss_list.csv", delimiter=",")
    plt.plot(epoch_list_for_the_plot, testing_loss_list, 'r')  # pass array or list
    plt.plot(epoch_list_for_the_plot, training_loss, 'g')
    plt.xlabel("No. of Epochs")
    plt.ylabel("Average Loss")
    # plt.legend(loc='upper left')
    # plt.gca().legend(('Training Loss', 'Testing Loss'))
    plt.title("Number of Epochs VS Loss")
    plt.grid()
    plt.show()


    #shuffle all dataset except the test sample, standardize using the values of the test sample as well
    # divide into train and test then do testing of the test sample along other test samples and print the last of the test sample i.e.,
    #alloy of interest

    #412

