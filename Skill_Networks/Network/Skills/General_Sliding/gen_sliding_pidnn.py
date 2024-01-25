from numpy.lib.npyio import save
import torch
import torch.autograd as autograd
from torch.autograd import grad
import torch.nn as nn

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# import friction_dataloader as mdl

optimizer = None

VT_u_train = None
X_u_train = None
VT_f_train = None
VT_test = None
X_test = None
VT_validation = None
X_validation = None

def set_loss(model, device, batch_size, data = None, X_true = None):
	# Returns MSE
	if data is None:
		data = VT_validation
		X_true = X_validation
	error = 0
	actual = 0
	size = X_true.shape[0]
	if size<1000: batch_size = size
	for i in range(0, size, batch_size):
		batch_data = data[i:min(i+batch_size,size), :]
		batch_X_true = X_true[i:min(i+batch_size,size), :]
		with torch.no_grad():
			batch_X_predicted = model.forward(batch_data)
		error += torch.sum((batch_X_predicted-batch_X_true)**2)/size
		actual += torch.sum((batch_X_true)**2)/size
	return error



def dataloader(config, device):

    data_df = pd.read_csv(config['MODEL_NAME'] + "data.csv", header=None)
    data = np.array(data_df)

    START_INPUT = 3
    END_INPUT = 5
    START_OUTPUT = 5
    END_OUTPUT = 7

    N_u = config['NUM_DATA']
    N_f = N_u * config['collocation_multiplier']

    global VT_u_train, X_u_train, VT_f_train, VT_test, X_test, VT_validation, X_validation

    VT_u_train = data[:N_u,START_INPUT:END_INPUT]
    X_u_train = data[:N_u,START_OUTPUT:END_OUTPUT]

    np.random.shuffle(data)
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

class PINN(nn.Module):
	
    def __init__(self, id, VT_u, XV_u, VT_f, layers, device, config, alpha, N_u, N_f):
        """
        id: Used for identifying the model during grid search
        VT_u: [Input for the data part] (Inittial velocity, Time)(VT), _u implies data points
        XV_u: [Ouput for the data part] Actual output, used during MSE calculation of the data part
        VT_f: [Input for the physics part] (Inittial velocity, Time)(VT), _f implies collocation points
        layers: layers[i] denotes number of neurons in ith layer
        alpha: hyperparameter
        N_u: Number of data points
        N_f: Number of collocation points
        config: YAML file data read is also passed in case any more parameters need to be read
        """

        super().__init__()

        self.id = id
        self.device = device

        self.alpha = alpha
        self.N_u = N_u
        self.N_f = N_f
        self.batch_size = config['BATCH_SIZE']
        self.config = config

        ## Unknown Mu Parameter
        self.mu = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        self.VT_u = VT_u
        self.XV_u = XV_u
        self.VT_f = VT_f
        self.layers = layers

        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0
        self.loss_u = torch.tensor(0)
        self.loss_f = torch.tensor(0)
        self.elapsed = None

        self.iter_history = []
        self.history = None # train, loss_u, loss_f, validation

        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=5/3)
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

    def batched_mse0(self, err):
        size = err.shape[0]
        if size < 1000: batch_size = size
        else: batch_size = self.batch_size
        mse = 0
        for i in range(0, size, batch_size):
            batch_err = err[i : min(i+batch_size,size)]
            mse += torch.sum((batch_err)**2)/size
        return mse

    def loss_DP(self,x,y):
        """ Loss at boundary and initial conditions """
        prediction = self.forward(x)
        error = prediction - y
        loss_u = self.batched_mse(error)
        return loss_u

    def loss_CP(self, VT_f_train):
        """ Loss at collocation points, calculated from Partial Differential Equation
        Note: x_v = x_v_t[:,[0]], x_t = x_v_t[:,[1]]
        """
                        
        g = VT_f_train.clone()  
        g.requires_grad = True

        u = self.forward(g)

        v = u[:,0]
        x = u[:,1]

        grads = torch.ones(v.shape, device=self.device)
        grad_x = grad(x, g, create_graph=True, grad_outputs=grads)[0]
        grad_v = grad(v, g, create_graph=True, grad_outputs=grads)[0]

        # calculate first order derivatives
        x_v = grad_x[:, 0]
        x_t = grad_x[:, 1]

        v_v = grad_v[:, 0]
        v_t = grad_v[:, 1]

        # calculate second order derivatives
        grad_x_t = grad(x_t, g, create_graph=True, grad_outputs=grads)[0]

        x_tt = grad_x_t[:, 1]

        f_x = (x_tt + self.mu * 9.8 * torch.ones(x_tt.shape, device=self.device))
        f_v = (v_t + self.mu * 9.8 * torch.ones(v_t.shape, device=self.device))

        loss_x = self.batched_mse0(f_x)
        loss_v = self.batched_mse0(f_v)

        loss_f = loss_x + loss_v

        return loss_f

    def loss(self,VT_u_train,X_u_train,VT_f_train):
        self.loss_u = self.loss_DP(VT_u_train,X_u_train)
        self.loss_f = self.alpha * self.loss_CP(VT_f_train)
        loss_val = self.loss_u + self.loss_f
        return loss_val

    def closure(self):
        """ Called multiple times by optimizers like Conjugate Gradient and LBFGS.
        Clears gradients, compute and return the loss.
        """
        optimizer.zero_grad()
        loss = self.loss(self.VT_u, self.XV_u, self.VT_f)
        loss.backward()		# To get gradients

        self.iter += 1

        if self.iter % 100 == 0:
            training_loss = loss.item()
            validation_loss = set_loss(self, self.device, self.batch_size).item()
            print(
                'Iter %d, Training: %.5e, Data loss: %.5e, Collocation loss: %.5e, Validation: %.5e' % (self.iter, training_loss, self.loss_u, self.loss_f, validation_loss)
            )
            print('Unknown Mu Value', self.mu.item())
            self.iter_history.append(self.iter)
            current_history = np.array([training_loss, self.loss_u.item(), self.loss_f.item(), validation_loss])
            if self.history is None: self.history = current_history.reshape(1,-1)
            else: self.history = np.vstack([self.history, current_history])
        return loss

def pidnn_driver(config):
    num_layers = config['num_layers']
    num_neurons = config['neurons_per_layer']

    torch.set_default_dtype(torch.float)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    device = torch.device('cuda' if torch.cuda.is_available() and config['CUDA_ENABLED'] else 'cpu')

    print("Running this on", device)
    if device == 'cuda': 
        print(torch.cuda.get_device_name())

    models = []
    validation_losses = []

    N_u = config['NUM_DATA']
    N_f = N_u * config['collocation_multiplier']

    dataloader(config, device)

    alpha = config['alpha']

    # layers is a list, not an ndarray
    layers = np.concatenate([[VT_u_train.shape[1]], num_neurons*np.ones(num_layers), [X_u_train.shape[1]]]).astype(int).tolist()


    print(f'++++++++++ N_u:{N_u}, N_f:{N_f}, Alpha:{alpha} ++++++++++')
        
    model = PINN((0,0), VT_u_train, X_u_train, VT_f_train, layers, device, config, alpha, N_u, N_f)
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
