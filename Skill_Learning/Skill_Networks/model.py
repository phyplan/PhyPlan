from numpy.lib.npyio import save
import torch
import torch.autograd as autograd
from torch.autograd import grad
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from data import dataloader
from loss import Loss

optimizer = None

VT_u_train = None
X_u_train = None
VT_f_train = None
VT_test = None
X_test = None
VT_validation = None
X_validation = None

np.random.seed(2103)
torch.manual_seed(2103)
 

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
        self.loss_func = Loss(config)
        self.loss = self.loss_func.loss
        self.val_loss_func = Loss(config)

        if ((config['SKILL'] == 'Sliding' or config['SKILL'] == 'Collision') and config['PARAMETER'] == 'Determine'):
            self.prm = torch.tensor(0.25, requires_grad=True, device = self.device)


    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).to(self.device)
                        
        x.requires_grad = True
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        if self.config['SKILL'] == 'Swinging':
            b = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + x[:, 1].reshape(-1, 1) # theta
            grads = torch.ones(b.shape, device=self.device)
            grad_x = grad(b, x, create_graph=True, grad_outputs=grads)[0]
            c = (grad_x[:, 0]).reshape(-1, 1) # omega
            return torch.cat([b, c], dim=1)
        elif self.config['SKILL'] == 'Sliding':
            if self.config['PARAMETER'] == 'Variable':
                b = (torch.log(torch.ones(x[:, 2].reshape(-1, 1).shape, device = self.device) + x[:, 2].reshape(-1, 1))) * torch.sin(x[:, 2].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + (x[:, 2]*x[:, 1]).reshape(-1, 1) # S
                grads = torch.ones(b.shape, device=self.device)
                grad_x = grad(b, x, create_graph=True, grad_outputs=grads)[0]
                c = (grad_x[:, 2]).reshape(-1, 1) # v
                return torch.cat([c, b], dim=1)
            else:
                b = a[:, 0].reshape(-1, 1) # S
                grads = torch.ones(b.shape, device=self.device)
                grad_x = grad(b, x, create_graph=True, grad_outputs=grads)[0]
                c = (grad_x[:, 1]).reshape(-1, 1) # v
                return torch.cat([c, b], dim=1) # Sliding
        elif self.config['SKILL'] == 'Throwing':
            b_z = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + (x[:, 0]*x[:, 1]).reshape(-1, 1) + x[:, 3].reshape(-1, 1) # z
            b_x = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 1].reshape(-1, 1) + (x[:, 0]*x[:, 2]).reshape(-1, 1) + x[:, 4].reshape(-1, 1) # x
            grads = torch.ones(b_z.shape, device=self.device)
            grad_x = grad(b_z, x, create_graph=True, grad_outputs=grads)[0]
            c_z = (grad_x[:, 0]).reshape(-1, 1) # v_z
            grads = torch.ones(b_x.shape, device=self.device)
            grad_x = grad(b_x, x, create_graph=True, grad_outputs=grads)[0]
            c_x = (grad_x[:, 0]).reshape(-1, 1) # v_x
            return torch.cat([b_z, c_z, b_x, c_x], dim=1) # Throwing
        else:
            return a

def evaluate_model(config, value, id = None):
    device = torch.device('cuda' if torch.cuda.is_available() and config['CUDA_ENABLED'] else 'cpu')
    if (config["LOAD_ONLY"]):
        model_eval = torch.load(config['LOAD_DIR'])
    else:
        if id != None:
            model_eval = torch.load(config["MODEL_DIR"] + config["MODEL_NAME"] + "_" + str(id) + ".pt")
        else:
            model_eval = torch.load(config["MODEL_DIR"] + config["MODEL_NAME"] + ".pt")
    if config['TRAIN_MODE'] == 'nn':
        X_train, Y_train, X_test, Y_test, X_validation, Y_validation = dataloader(device,value, config)
        pred = model_eval.forward(X_validation)
        mse = nn.functional.mse_loss(Y_validation, pred)
        rel_l2 = torch.norm(torch.flatten(pred - Y_validation), p=2, dim=-1, keepdim=False)/torch.norm(torch.flatten(Y_validation), p=2, dim=-1, keepdim=False)
        rel_l2_all = torch.norm(pred - Y_validation, p=2, dim=0, keepdim=False)/torch.norm(Y_validation, p=2, dim=0, keepdim=False)
        print("MSE Loss: {}".format(mse))
        print("Relative L2 Error: {}".format(rel_l2))
        print("Relative L2 Error in each component: {}".format(rel_l2_all.detach().numpy()))
        if config["SAVE_EVAL"]:
            with open(config['EVAL_OUT'], 'a+') as eval_file:
                np.savetxt(eval_file, torch.cat([X_validation, Y_validation, pred], 1).cpu().detach().numpy())
        return torch.cat([torch.tensor([mse, rel_l2]), rel_l2_all]).detach().numpy()

    else:
        VT_u_train, X_u_train, VT_f_train, VT_test, X_test, VT_validation, X_validation = dataloader(device,value, config)
        pred = model_eval.forward(VT_validation)
        mse = nn.functional.mse_loss(X_validation, pred)
        rel_l2 = torch.norm(torch.flatten(pred - X_validation), p=2, dim=-1, keepdim=False)/torch.norm(torch.flatten(X_validation), p=2, dim=-1, keepdim=False)
        rel_l2_all = torch.norm(pred - X_validation, p=2, dim=0, keepdim=False)/torch.norm(X_validation, p=2, dim=0, keepdim=False)
        print("MSE Loss: {}".format(mse))
        print("Relative L2 Error: {}".format(rel_l2))
        print("Relative L2 Error in each component: {}".format(rel_l2_all.detach().numpy()))
        if config["SAVE_EVAL"]:
            with open(config['EVAL_OUT'], 'w+') as eval_file:
                np.savetxt(eval_file, torch.cat([VT_validation, X_validation, pred], 1).cpu().detach().numpy())
        if (config["SKILL"] == 'Sliding' or config["SKILL"] == 'Collision') and config["PARAMETER"] == 'Determine':
            print("Learnt prm: {}".format(model_eval.prm.detach().numpy()))
            return torch.cat([torch.tensor([mse, rel_l2]), rel_l2_all]).detach().numpy(), model_eval.prm.detach().numpy()
        else:
            return torch.cat([torch.tensor([mse, rel_l2]), rel_l2_all]).detach().numpy()

def train_model(config,value, id=None):

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
    N_f = config['NUM_COLL']

    alpha = config['alpha']

    if config['TRAIN_MODE'] == 'nn':
        X_train, Y_train, X_test, Y_test, X_validation, Y_validation = dataloader(device,value, config)
        layers = np.concatenate([[X_train.shape[1]], num_neurons*np.ones(num_layers), [Y_train.shape[1]]]).astype(int).tolist()
        model = Skill_Network((0,0), X_train, Y_train, layers, device, config, alpha, N_u, N_f,nn.Tanh,None)
        model.VT_validation = X_validation
        model.X_validation = Y_validation

    else:
        VT_u_train, X_u_train, VT_f_train, VT_test, X_test, VT_validation, X_validation = dataloader(device,value, config)
        if config['SKILL'] == 'Collision':
            layers = np.concatenate([[VT_u_train.shape[1]], num_neurons*np.ones(num_layers), [X_u_train.shape[1]]]).astype(int).tolist()
            model = Skill_Network((0,0), VT_u_train, X_u_train, layers, device, config, alpha, N_u, N_f,nn.Tanh,VT_f_train)
        else:
            layers = np.concatenate([[VT_u_train.shape[1]], num_neurons*np.ones(num_layers), [X_u_train.shape[1]/2]]).astype(int).tolist() # Autograd Integration in Outputs
            model = Skill_Network((0,0), VT_u_train, X_u_train, layers, device, config, alpha, N_u, N_f,nn.Tanh,VT_f_train)
        model.VT_validation = VT_validation
        model.X_validation = X_validation


    model.to(device)

    mode = config['TRAIN_MODE']
    
    print(f'++++++++++ Train_Mode:{mode}, N_u:{N_u}, N_f:{N_f}, Alpha:{alpha} ++++++++++')


    global optimizer
    # Adam Optimizer
    if (config['SKILL'] == 'Sliding' or config['SKILL'] == 'Collision') and config['PARAMETER'] == 'Determine':
        l = list(model.parameters())
        l.append(model.prm)
        optimizer1 = torch.optim.Adam(params=model.parameters(), lr=6.4e-4)
        optimizer2 = torch.optim.Adam([model.prm], lr=2.1e-3)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 210, gamma=0.978, last_epoch=-1)
    start_time = time.time()
    for i in range(config['EARLY_STOPPING']):
        if (config["SKILL"] == 'Sliding' or config['SKILL'] == 'Collision') and config["PARAMETER"] == "Determine":
            optimizer1.zero_grad()
            optimizer2.zero_grad()
        else:
            optimizer.zero_grad()
        if config["TRAIN_MODE"] == 'pinn':
            loss = model.loss(model, model.VT_u, model.XV_u, model.VT_f)
        else:
            loss = model.loss(model, model.VT_u, model.XV_u)
        loss.backward()

        model.iter += 1
        if model.iter % 100 == 0:
            training_loss = loss.item()
            validation_loss = model.val_loss_func.set_loss(model, model.device, model.batch_size,model.VT_validation,model.X_validation).item()
            print(
                'Iter %d, Training: %.5e, Data loss: %.5e, Collocation loss: %.5e, Validation: %.5e' % (model.iter, training_loss, model.loss_func.loss_u, model.loss_func.loss_f, validation_loss)
            )
        if (config["SKILL"] == 'Sliding' or config['SKILL'] == 'Collision') and config["PARAMETER"] == "Determine":
            optimizer1.step()
            optimizer2.step()
        else:
            optimizer.step()
            scheduler.step()
    elapsed = time.time() - start_time    
    print('Training time: %.2f' % (elapsed))

    """ Saving only final model for reloading later """
    if config['SAVE_MODEL']: 
        if id != None:
            torch.save(model, config['MODEL_DIR'] + config['MODEL_NAME'] + "_" + str(id) + '.pt')
        else:
            torch.save(model, config['MODEL_DIR'] + config['MODEL_NAME'] + '.pt')

    if device == 'cuda':
        torch.cuda.empty_cache()