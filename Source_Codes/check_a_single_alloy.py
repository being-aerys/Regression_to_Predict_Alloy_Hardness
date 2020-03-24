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

    '''Create a neural network'''
    network = NeuralNetwork(13,1)

    '''Load the saved model's parameters'''
    saved_state_statistics_of_the_model = torch.load('../Saved_Models/Melting_Point_412.pth')
    for keyname_of_the_state_statistic in saved_state_statistics_of_the_model:
        print(keyname_of_the_state_statistic)

    '''Get your new model's statistics'''
    new_model_statistics_dictionary = network.state_dict()  # get the model's statistics first

    '''Override the statistics varible's parameters with the ones from the saved model'''
    for key, value in saved_state_statistics_of_the_model.items():
        if key in new_model_statistics_dictionary:
            new_model_statistics_dictionary.update({key: value})

    '''load these new parameters onto your new model'''
    network.load_state_dict(new_model_statistics_dictionary)

    '''Keep the model in eval mode since you have a dropout layer'''
    network.eval()


    #Order in data set: Cr Hf Mo Nb Ta Ti Re V   W  Zr Co Ni Fe Al Mn Cu C En  Bulk shear  Hardness Valence MeltingTemp
    #                         10 20 30    15     20                              246 106   979      5.65     2956

    input =[0,0,10,20,30,0,15,0,20,0,0,0,0,0,0,0,0,576,5.65]

    input = torch.tensor(input)

    output = network(input)

    print("The melting point of this alloy is: ", output)

    #412

    print("Then mean values for all the features are: ", scaler.mean_)
    print("\nThe variances for all the eatures are: ", scaler.var_)




