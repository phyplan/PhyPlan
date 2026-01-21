from numpy.lib.npyio import save
import torch
import torch.autograd as autograd
from torch.autograd import grad
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def dataloader(device, val, config):
    
    data_df = pd.read_csv(config['DATASET'], header=None)
    if (config['SKILL'] == 'Collision'):
        data_df = pd.read_csv(config['DATASET'])

    data = np.array(data_df)

    if (config['TRAIN_MODE'] == 'pinn'):
        coll_df = pd.read_csv(config['COLL'], header=None)
        if (config['SKILL'] == 'Collision'):
            coll_df = pd.read_csv(config['COLL'])

        coll = np.array(coll_df)

    data_val = pd.read_csv(config['DATA_VAL'], header=None)
    if (config['SKILL'] == 'Collision'):
        data_val = pd.read_csv(config['DATA_VAL'])

    data_v = np.array(data_val)


    if not(config['EVALUATE']):
        data_v = data_v[np.random.choice(len(data_v), size=int(len(data_v)), replace=False)]

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

        X_validation = torch.from_numpy(data_v[:,START_INPUT:END_INPUT]).float().to(device)
        Y_validation = torch.from_numpy(data_v[:,START_OUTPUT:END_OUTPUT]).float().to(device)
        X_test = X
        Y_test = Y

        return X_train, Y_train, X_test, Y_test, X_validation, Y_validation


    else:
        N_f = config['NUM_COLL']

        VT_u_train = data[:N_u,START_INPUT:END_INPUT]
        X_u_train = data[:N_u,START_OUTPUT:END_OUTPUT]

     
        VT_f_train = coll[:N_f,START_INPUT:END_INPUT]

        VT_u_train = torch.from_numpy(VT_u_train).float().to(device)
        X_u_train = torch.from_numpy(X_u_train).float().to(device)
        VT_f_train = torch.from_numpy(VT_f_train).float().to(device)

        VT = data[:,START_INPUT:END_INPUT]
        VT = torch.from_numpy(VT).float().to(device)
        X = data[:,START_OUTPUT:END_OUTPUT]
        X = torch.from_numpy(X).float().to(device)

        VT_validation = torch.from_numpy(data_v[:,START_INPUT:END_INPUT]).float().to(device)
        X_validation = torch.from_numpy(data_v[:,START_OUTPUT:END_OUTPUT]).float().to(device)
        VT_test = VT
        X_test = X

        return VT_u_train, X_u_train, VT_f_train, VT_test, X_test, VT_validation, X_validation