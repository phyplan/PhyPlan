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


class Loss():
    def __init__(self): 
        self.alpha = config['alpha']
        self.batch_size = config['BATCH_SIZE']
        self.loss_f = 0
        self.loss_u = 0

    def set_loss(self,model, device, batch_size, data = None, X_true = None):
            # Returns MSE
        if data is None:
            data = X_validation
            X_true = Y_validation
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

    def batched_mse(self,err):
        size = err.shape[0]
        if size < 1000: batch_size = size
        else: batch_size = self.batch_size
        mse = 0
        for i in range(0, size, batch_size):
            batch_err = err[i : min(i+batch_size,size), :]
            mse += torch.sum((batch_err)**2)/size
        return mse

    def batched_mse0(self,err):
        size = err.shape[0]
        if size < 1000: batch_size = size
        else: batch_size = self.batch_size
        mse = 0
        for i in range(0, size, batch_size):
            batch_err = err[i : min(i+batch_size,size)]
            mse += torch.sum((batch_err)**2)/size
        return mse

    def loss_DP(self,model,x,y):
        """ Loss at boundary and initial conditions """
        prediction = model.forward(x)
        error = prediction - y
        loss_u = self.batched_mse(error)
        return loss_u

    def loss_CP(self,model,VT_f_train):
        """ Loss at collocation points, calculated from Partial Differential Equation
        Note: x_v = x_v_t[:,[0]], x_t = x_v_t[:,[1]]
        """   
        g = VT_f_train.clone()  
        g.requires_grad = True

        u = model.forward(g)

        if config['SKILL'] == 'Swinging':

            cur_theta = u[:,0]
            cur_omega = u[:,1]

            grads = torch.ones(cur_theta.shape, device=model.device)
            grad_x = grad(cur_theta, g, create_graph=True, grad_outputs=grads)[0]

            # calculate first order derivatives
            theta_t = grad_x[:, 1]
            grad_x = grad(theta_t, g, create_graph=True, grad_outputs=grads)[0]
            theta_tt = grad_x[:, 1]

            f1 = (theta_tt + 9.8 * torch.sin(cur_theta)) 

            grad_x = grad(cur_omega, g, create_graph=True, grad_outputs=grads)[0]
            omega_t = grad_x[:, 1]

            f2 = (omega_t + 9.8 * torch.sin(cur_theta)) 
            loss_v = self.batched_mse0(f2)

            loss_x = self.batched_mse0(f1)
            
            loss_f = loss_x + loss_v


        elif config['SKILL'] == 'Sliding':

            v = u[:,0]
            x = u[:,1]

            grads = torch.ones(v.shape, device=model.device)
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

            if config['PARAMETER'] == 'Fixed':
                f_x = (x_tt + 0.2 * 9.8 * torch.ones(x_tt.shape, device=model.device))
                f_v = (v_t + 0.2 * 9.8 * torch.ones(v_t.shape, device=model.device))
                
            elif config['PARAMETER'] == 'Determine':
                f_x = (x_tt + model.mu * 9.8 * torch.ones(x_tt.shape, device=model.device))
                f_v = (v_t + model.mu * 9.8 * torch.ones(v_t.shape, device=model.device))
            else: 
                f_x = (x_tt + g[:,0] * 9.8 )
                f_v = (v_t + g[:,0] * 9.8) 

            loss_x = self.batched_mse0(f_x)
            loss_v = self.batched_mse0(f_v)

            loss_f = loss_x + loss_v

        
        elif config['SKILL'] == 'Throwing':

            vy = u[:,0]
            y = u[:,1]
            x = u[:,2]
        
            grads = torch.ones(x.shape, device=model.device)
            grad_vy = grad(vy, g, create_graph=True, grad_outputs=grads)[0]
            grad_y = grad(y, g, create_graph=True, grad_outputs=grads)[0]
            grad_x = grad(x, g, create_graph=True, grad_outputs=grads)[0]

            x_t = grad_x[:, 0]
            y_t = grad_y[:, 0]
            vy_t = grad_vy[:, 0]



            grad_y_t = grad(y_t, g, create_graph=True, grad_outputs=grads)[0]
            grad_x_t = grad(x_t, g, create_graph=True, grad_outputs=grads)[0]


            x_tt = grad_x_t[:, 0]
            y_tt = grad_y_t[:, 0]

            f_x = (x_tt)
            f_y = (y_tt + 9.8)
            f_vy = (vy_t + 9.8)
            
            loss_x = self.batched_mse0(f_x)
            loss_y = self.batched_mse0(f_y)
            loss_vy = self.batched_mse0(f_vy)

            loss_f = loss_x + loss_y + loss_vy

        else:

            loss_f =  None

        return loss_f

    def loss(self,model,X,Y,collocation = None):
        
        self.loss_u = self.loss_DP(model,X,Y)
        if config["TRAIN_MODE"] == "pinn":
            self.loss_f = self.alpha * self.loss_CP(model,collocation)
            loss_val = self.loss_f + self.loss_u
        else:
            loss_val = self.loss_u
        return loss_val
