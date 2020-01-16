#Mo15Nb20Re15Ta30W20 2850

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

    '''Create a neural network'''
    network = NeuralNetwork(15,1)

    '''Load the saved model's parameters'''
    saved_state_statistics_of_the_model = torch.load("mytraining2.pth")
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

    # Mo15 Nb20 Re15 Ta30 W20 --> 2850

    #Cr Hf Mo Nb Ta Ti Re V W Zr TransferTemp VolumePer EnergyPer Binding Density BindingEnergy , Melting Temp
    input, label = [0,0,15,20,30,0,15,0,20,0,878,11.8263,],0

    output = network(input)

    print("The melting point of this alloy is: ", output)



