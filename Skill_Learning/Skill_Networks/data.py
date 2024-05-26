from numpy.lib.npyio import save
import torch
import torch.autograd as autograd
from torch.autograd import grad
import torch.nn as nn
import yaml
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

config_filename = "config.yaml"

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)



def dataloader(device, val):
    
    data_df = pd.read_csv(config['DATASET'], header=None)
    data = np.array(data_df)
    data = np.array(data_df) 
    data = data[np.random.choice(len(data), size=int(len(data)), replace=False)]

    START_INPUT = val[0]
    END_INPUT = val[1]
    START_OUTPUT = val[2]
    END_OUTPUT = val[3]

    N_u = config['NUM_DATA']

    if config['TRAIN_MODE'] == 'nn':

        X_train = data[:N_u,START_INPUT:END_INPUT]
        Y_train = data[:N_u,START_OUTPUT:END_OUTPUT]

        X_train = torch.from_numpy(X_train).float().to(device)
        Y_train = torch.from_numpy(Y_train).float().to(device)

        X = data[:,START_INPUT:END_INPUT]
        Y = data[:,START_OUTPUT:END_OUTPUT]
        X = torch.from_numpy(X).float().to(device)
        Y = torch.from_numpy(Y).float().to(device)

        X_validation = X
        Y_validation = Y
        X_test = X
        Y_test = Y

        return X_train, Y_train, X_test, Y_test, X_validation, Y_validation


    else:
        N_f = N_u * config['collocation_multiplier']

        VT_u_train = data[:N_u,START_INPUT:END_INPUT]
        X_u_train = data[:N_u,START_OUTPUT:END_OUTPUT]

     
        VT_f_train = data[:N_f,START_INPUT:END_INPUT]

        VT_u_train = torch.from_numpy(VT_u_train).float().to(device)
        X_u_train = torch.from_numpy(X_u_train).float().to(device)
        VT_f_train = torch.from_numpy(VT_f_train).float().to(device)

        VT = data[:,START_INPUT:END_INPUT]
        VT = torch.from_numpy(VT).float().to(device)
        X = data[:,START_OUTPUT:END_OUTPUT]
        X = torch.from_numpy(X).float().to(device)

        VT_validation = VT
        X_validation = X
        VT_test = VT
        X_test = X

        return VT_u_train, X_u_train, VT_f_train, VT_test, X_test, VT_validation, X_validation
        
 






    

    