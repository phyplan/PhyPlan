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
from data import dataloader
from loss import Loss

 
config_filename = "config.yaml"

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)


optimizer = None

VT_u_train = None
X_u_train = None
VT_f_train = None
VT_test = None
X_test = None
VT_validation = None
X_validation = None

 

class Skill_Network(nn.Module):
	
    def __init__(self, id, VT_u, XV_u, layers, device, config, alpha, N_u, N_f,act,VT_f= None):
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

        self.VT_u = VT_u
        self.XV_u = XV_u
        self.VT_f = VT_f
        self.layers = layers

        self.activation = act()
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        for i in range(len(layers)-1):
            
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)
            

        self.iter = 0
        self.loss_u = torch.tensor(0)
        self.loss_f = torch.tensor(0)
        self.elapsed = None
        self.loss_func = Loss()
        self.loss = self.loss_func.loss
        self.val_loss_func = Loss()

         
            
            

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
         
         
        

    def closure(self):
        """ Called multiple times by optimizers like Conjugate Gradient and LBFGS.
        Clears gradients, compute and return the loss.
        """
        optimizer.zero_grad()
        if config["TRAIN_MODE"] == 'pinn':
            loss = self.loss(self,self.VT_u, self.XV_u, self.VT_f)
        else:
            loss = self.loss(self,self.VT_u, self.XV_u)
        loss.backward()		# To get gradients

        self.iter += 1

        if self.iter % 100 == 0:
            training_loss = loss.item()
            validation_loss = self.val_loss_func.set_loss(self, self.device, self.batch_size,self.VT_validation,self.X_validation).item()
            print(
                'Iter %d, Training: %.5e, Data loss: %.5e, Collocation loss: %.5e, Validation: %.5e' % (self.iter, training_loss, self.loss_func.loss_u, self.loss_func.loss_f, validation_loss)
            )
        return loss

def train_model(config,value):

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

    alpha = config['alpha']

    # layers is a list, not an ndarray

    


    if config['TRAIN_MODE'] == 'nn':
        X_train, Y_train, X_test, Y_test, X_validation, Y_validation = dataloader(device,value)
        layers = np.concatenate([[X_train.shape[1]], num_neurons*np.ones(num_layers), [Y_train.shape[1]]]).astype(int).tolist()
        model = Skill_Network((0,0), X_train, Y_train, layers, device, config, alpha, N_u, N_f,nn.Tanh,None)
        model.VT_validation = X_validation
        model.X_validation = Y_validation

    else:
        VT_u_train, X_u_train, VT_f_train, VT_test, X_test, VT_validation, X_validation = dataloader(device,value)
        layers = np.concatenate([[VT_u_train.shape[1]], num_neurons*np.ones(num_layers), [X_u_train.shape[1]]]).astype(int).tolist()
        model = Skill_Network((0,0), VT_u_train, X_u_train, layers, device, config, alpha, N_u, N_f,nn.Tanh,VT_f_train)
        model.VT_validation = VT_validation
        model.X_validation = X_validation


    model.to(device)

    mode = config['TRAIN_MODE']
    
    print(f'++++++++++ Train_Mode:{mode}, N_u:{N_u}, N_f:{N_f}, Alpha:{alpha} ++++++++++')


    # L-BFGS Optimizer
    global optimizer
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.01, 
        tolerance_grad= 1 * np.finfo(float).eps,
        tolerance_change= 1 * np.finfo(float).eps,
        max_iter = config['EARLY_STOPPING'],
        history_size = 100
    )
    
    start_time = time.time()
    optimizer.step(model.closure)		 
    elapsed = time.time() - start_time                
    print('Training time: %.2f' % (elapsed))

    # validation_loss = model.val_loss_func.set_loss(model, device, config['BATCH_SIZE'])
    # model.elapsed = elapsed
    # model.to('cpu')
    # models.append(model)
    # validation_losses.append(validation_loss.cpu().item())

    model_id = 0 #np.nanargmin(validation_losses)  
    #model = models[model_id]

    # """ Model Accuracy """ 
    # error_validation = validation_losses[model_id]
    # print('Validation Error of finally selected model: %.5f'  % (error_validation))

    """ Saving only final model for reloading later """
    if config['SAVE_MODEL']: torch.save(model, config['MODEL_DIR'] + config['MODEL_NAME'] + '.pt')

    if device == 'cuda':
        torch.cuda.empty_cache()
