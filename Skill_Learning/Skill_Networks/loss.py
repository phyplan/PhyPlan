from numpy.lib.npyio import save
import torch
import torch.autograd as autograd
from torch.autograd import grad
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class Loss():
    def __init__(self, config): 
        self.config = config
        self.alpha = config['alpha']
        self.batch_size = config['BATCH_SIZE']
        self.loss_f = 0
        self.loss_u = 0

    def set_loss(self,model, device, batch_size, data, X_true):
        return self.loss_DP(model,data,X_true, "validation")

    def batched_mse(self,err):
        size = err.shape[0]
        batch_size = self.config["BATCH_SIZE"]
        mse = 0
        for i in range(0, size, batch_size):
            batch_err = err[i : min(i+batch_size,size), :]
            mse += torch.sum((batch_err)**2)/(size*err.shape[1])
        return mse

    def batched_mse0(self,err):
        size = err.shape[0]
        batch_size = self.config["BATCH_SIZE"]
        mse = 0
        for i in range(0, size, batch_size):
            batch_err = err[i : min(i+batch_size,size)]
            mse += torch.sum((batch_err)**2)/size
        return mse

    def loss_DP(self,model,x,y, mode):
        """ Loss at boundary and initial conditions """
        prediction = model.forward(x)
        error = (prediction - y)
        if self.config['SKILL'] == "Throwing":
            if mode == 'train':
                error = error[:, [0, 2]]
                y = y[:, [0, 2]]
            else:
                pass

        elif self.config['SKILL'] == "Sliding":
            if mode == 'train':
                error = error[:, 1].reshape(-1, 1)
                y = y[:, 1].reshape(-1, 1)
            else:
                pass
        elif self.config['SKILL'] == "Swinging":
            if mode == 'train':
                error = error[:, 0].reshape(-1, 1)
                y = y[:, 0].reshape(-1, 1)
            else:
                pass
        if mode=='train':
            loss_u = self.batched_mse(error)
        else:
            loss_u = torch.norm(torch.flatten(error), p=2, dim=-1, keepdim=False)/torch.norm(torch.flatten(y), p=2, dim=-1, keepdim=False)
        return loss_u

    def loss_CP(self,model,VT_f_train):
        """ Loss at collocation points, calculated from Partial Differential Equation
        Note: x_v = x_v_t[:,[0]], x_t = x_v_t[:,[1]]
        """   
        g = VT_f_train.clone()  
        g_0 = g.clone().to(model.device)
        g.requires_grad = True

        u = model.forward(g)

        if self.config['SKILL'] == 'Swinging':
            cur_theta = u[:,0]
            cur_omega = u[:,1]

            grads = torch.ones(cur_theta.shape, device=model.device)
            grad_x = grad(cur_theta, g, create_graph=True, grad_outputs=grads)[0]

            # calculate first order derivatives
            theta_t = grad_x[:, 0]
            grad_x = grad(theta_t, g, create_graph=True, grad_outputs=grads)[0]
            theta_tt = grad_x[:, 0]

            f1 = (theta_tt + 9.8 * torch.sin(cur_theta))
            loss_x = self.batched_mse0(f1)
            
            loss_f = loss_x


        elif self.config['SKILL'] == 'Sliding':
            if self.config['PARAMETER'] == 'Variable':
                v = u[:,0]
                x = u[:,1]

                grads = torch.ones(v.shape, device=model.device)
                grad_v = grad(v, g, create_graph=True, grad_outputs=grads)[0]
            else:
                g_0[:, 1] = torch.zeros(g_0[:, 0].shape, device=model.device)
                u_0 = model.forward(g_0)
                v = u[:,0]
                x = u[:,1]

                grads = torch.ones(v.shape, device=model.device)
                grad_v = grad(v, g, create_graph=True, grad_outputs=grads)[0]

                x_tt = grad_v[:, 1]
            if self.config['PARAMETER'] == 'Determine':
                f_x = (x_tt + (model.prm) * 9.8 * torch.ones(x_tt.shape, device=model.device))
                f_0 = u_0 - torch.cat([(g_0[:, 0]).reshape(-1, 1), torch.zeros(g_0[:, 0].shape, device=model.device).reshape(-1, 1)], dim = 1)

            else: 
                f_x = (x_tt + g[:,0] * 9.8 * torch.ones(x_tt.shape, device=model.device))

            loss_x = self.batched_mse0(f_x)
            loss_f = loss_x
            if self.config['PARAMETER'] != 'Variable':
                loss_0 = self.batched_mse0(f_0)
                loss_f += loss_0
        
        elif self.config['SKILL'] == 'Throwing':
            vy = u[:,1]
            y = u[:,0]
            x = u[:,2]
            vx = u[:, 3]
            grads = torch.ones(x.shape, device=model.device)
            grad_vy = grad(vy, g, create_graph=True, grad_outputs=grads)[0]
            grad_vx = grad(vx, g, create_graph=True, grad_outputs=grads)[0]

            x_tt = grad_vx[:, 0]
            y_tt = grad_vy[:, 0]

            f_x = (x_tt)
            f_y = (y_tt + 9.8)
            loss_x = self.batched_mse0(f_x)
            loss_y = self.batched_mse0(f_y)
            loss_f = loss_x + loss_y
        
        elif self.config['SKILL'] == 'Collision':

            v2 = u[:,1]
            v1 = u[:,0]
            if self.config["PARAMETER"] == 'Determine':
                coeff_e = model.prm
                u1 = g[:,1]
                m1 = g[:,0]
            else:
                coeff_e = g[:,0]
                u1 = g[:,2]
                m1 = g[:,1]
            loss_m = self.batched_mse0(m1*v1 + (1-m1)*v2 - m1*u1)
            loss_e = self.batched_mse0(v2 - v1 - coeff_e*u1)
            loss_f = loss_m + loss_e

        return loss_f

    def loss(self,model,X,Y,collocation = None):
        if not(self.config['PHYSICS_ONLY']):
            self.loss_u = self.loss_DP(model,X,Y,'train')
        else:
            self.loss_u = 0
        if self.config["TRAIN_MODE"] == "pinn":
            self.loss_f = self.loss_CP(model,collocation)
            loss_val = (self.alpha * self.loss_f + self.loss_u)/(1 + self.alpha)
        else:
            loss_val = self.loss_u
        return loss_val