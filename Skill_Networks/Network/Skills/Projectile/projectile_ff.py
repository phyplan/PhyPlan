from numpy.lib.npyio import save
import torch
import torch.autograd as autograd
from torch.autograd import grad
import torch.nn as nn

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

optimizer = None

X_train = None
Y_train = None

X_test = None
Y_test = None
X_validation = None
Y_validation = None

def set_loss(model, device, batch_size, data = None, Y_true = None):
	# Returns MSE
	if data is None:
		data = X_validation
		Y_true = Y_validation
	error = 0
	actual = 0
	size = Y_true.shape[0]
	if size<1000: batch_size = size
	for i in range(0, size, batch_size):
		batch_data = data[i:min(i+batch_size,size), :]
		batch_Y_true = Y_true[i:min(i+batch_size,size), :]
		with torch.no_grad():
			batch_Y_predicted = model.forward(batch_data)
		error += torch.sum((batch_Y_predicted-batch_Y_true)**2)/size
		actual += torch.sum((batch_Y_true)**2)/size
	return error


def dataloader(config, device):

    data_df = pd.read_csv(config['MODEL_NAME'] + "data.csv", header=None)
    data = np.array(data_df)

    START_INPUT = 2
    END_INPUT = 5
    START_OUTPUT = 5
    END_OUTPUT = 8

    N_u = config['NUM_DATA']

    global X_train, Y_train, X_test, Y_test, X_validation, Y_validation

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

class FF_Baseline(nn.Module):
	
    def __init__(self, id, X, Y, layers, device, config, N_u):
        """
        id: Used for identifying the model during grid search
        X: [Input for the data part] Implies data points
        Y: [Ouput for the data part] Actual output, used during MSE calculation of the data part
        layers: layers[i] denotes number of neurons in ith layer
        N_u: Number of data points
        config: YAML file data read is also passed in case any more parameters need to be read
        """

        super().__init__()

        self.id = id
        self.device = device

        self.N_u = N_u
        self.batch_size = config['BATCH_SIZE']
        self.config = config

        self.X = X
        self.Y = Y
        self.layers = layers

        self.activation = nn.ReLU()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0
        self.loss_u = None
        self.elapsed = None

        self.iter_history = []
        self.history = None

        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).to(self.device)
                        
        a = x.float()

        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)

        # Activation is not applied to last layer
        a = self.linears[-1](a)

        return a

    def batched_mse(self, err):

        size = err.shape[0]
        if size < 1000: batch_size = size
        else: batch_size = self.batch_size
        mse = 0

        for i in range(0, size, batch_size):
            batch_err = err[i : min(i+batch_size,size), :]
            mse += torch.sum((batch_err)**2)/size

        return mse


    def loss_DP(self,x,y):
        """ Loss at Data Points"""
        prediction = self.forward(x)
        error = prediction - y
        loss_u = self.batched_mse(error)
        return loss_u

    def loss(self,X_u_train,Y_u_train):
        """ Total Loss """
        self.loss_u = self.loss_DP(X_u_train,Y_u_train)
        loss = self.loss_u
        return loss

    def closure(self):
        """ Called multiple times by optimizers like Conjugate Gradient and LBFGS.
        Clears gradients, compute and return the loss.
        """
        optimizer.zero_grad()
        loss = self.loss(self.X, self.Y)
        loss.backward()		# To get gradients

        self.iter += 1

        if self.iter % 100 == 0:
            training_loss = loss.item()
            validation_loss = set_loss(self, self.device, self.batch_size).item()
            # training_history[self.id].append([self.iter, training_loss, validation_loss])
            print(
                'Iter %d, Training Loss: %.5e, Validation Loss: %.5e' % (self.iter, training_loss, validation_loss)
            )
            self.iter_history.append(self.iter)
            current_history = np.array([training_loss, validation_loss])
            if self.history is None: self.history = current_history.reshape(1,-1)
            else: self.history = np.vstack([self.history, current_history])

        return loss

def ff_driver(config):

    num_layers = config['num_layers']
    num_neurons = config['neurons_per_layer']
    torch.set_default_dtype(torch.float)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() and config['CUDA_ENABLED'] else 'cpu')

    print("Running this on", device)

    if device == 'cuda': 
        print(torch.cuda.get_device_name())

    dataloader(config, device)

    # layers is a list, not an ndarray
    layers = np.concatenate([[X_train.shape[1]], num_neurons*np.ones(num_layers), [Y_train.shape[1]]]).astype(int).tolist()

    models = []
    validation_losses = []

    N_u = config['NUM_DATA'] 

    model = FF_Baseline((0,0), X_train, Y_train, layers, device, config, N_u)
    model.to(device)

    # L-BFGS Optimizer
    global optimizer
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.01, 
        max_iter = config['EARLY_STOPPING'],
        tolerance_grad = 1.0 * np.finfo(float).eps, 
        tolerance_change = 1.0 * np.finfo(float).eps, 
        history_size = 100
    )

    start_time = time.time()
    optimizer.step(model.closure)		# Does not need any loop like Adam
    elapsed = time.time() - start_time                
    print('Training time: %.2f' % (elapsed))

    validation_loss = set_loss(model, device, config['BATCH_SIZE'])
    model.elapsed = elapsed
    model.to('cpu')
    models.append(model)
    validation_losses.append(validation_loss.cpu().item())

    model_id = np.nanargmin(validation_losses) # choosing best model out of the bunch
    model = models[model_id]

    """ Model Accuracy """ 
    error_validation = validation_losses[model_id]
    print('Validation Error of finally selected model: %.5f'  % (error_validation))

    """ Saving only final model for reloading later """
    if config['SAVE_MODEL']: torch.save(model, config['MODEL_DIR'] + config['MODEL_NAME'] + '.pt')

    if device == 'cuda':
        torch.cuda.empty_cache()