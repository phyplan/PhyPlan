import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import grad

class Skill_Network(nn.Module):
	
    def __init__(self, layers, device) -> None:
        """
        layers: layers[i] denotes number of neurons in ith layer
        """

        super().__init__()

        self.device = device
        self.layers = layers

        self.activation = nn.Tanh()
        # self.activation = act()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x, physics, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if device is not None:
            self.device = device

        if not(torch.is_tensor(x)):
            if physics in ['swinging', 'throwing', 'sliding', 'rolling']:
                step_size = x[2] if (physics == 'sliding' or physics == 'rolling') else x[0]
                lower_limit = step_size
                upper_limit = 1.0 + step_size
                range_col = torch.arange(lower_limit, upper_limit + step_size, step=step_size).unsqueeze(1)
                N = range_col.shape[0]
                if not(physics == 'sliding' or physics == 'rolling'):
                    x = torch.cat([range_col, torch.from_numpy(np.array(x)[1:]).repeat(N, 1)], dim=1).to(self.device)
                else:
                    x = torch.cat([torch.from_numpy(np.array(x)[:2]).repeat(N, 1), range_col], dim=1).to(self.device)
            else:
                x = torch.from_numpy(np.array(x)).to(self.device)
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.reshape(1, -1)
        

        if not x.requires_grad:
            x.requires_grad = True
        
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)

        # Activation is not applied to last layer
        a = self.linears[-1](a)
        if physics == 'swinging':
            # b = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + x[:, 1].reshape(-1, 1)
            b = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + (x[:, 0]*x[:, 2]).reshape(-1, 1) + x[:, 1].reshape(-1, 1) # theta # 20240830
            grads = torch.ones(b.shape, device=self.device)
            grad_x = grad(b, x, create_graph=True, grad_outputs=grads)[0]
            c = (grad_x[:, 0]).reshape(-1, 1) # omega
            return torch.cat([b, c], dim=1) # Swinging
        elif physics == 'sliding':
            b = (torch.log(torch.ones(x[:, 2].reshape(-1, 1).shape, device = self.device) + x[:, 2].reshape(-1, 1))) * torch.sin(x[:, 2].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + (x[:, 2]*x[:, 1]).reshape(-1, 1) # S # 20240830
            grads = torch.ones(b.shape, device=self.device)
            grad_x = grad(b, x, create_graph=True, grad_outputs=grads)[0]
            c = (grad_x[:, 2]).reshape(-1, 1) # v
            return torch.cat([c, b], dim=1) # Sliding
        elif physics == 'throwing':
            b_z = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + (x[:, 0]*x[:, 1]).reshape(-1, 1) #+ x[:, 3].reshape(-1, 1) # z # 20240830
            b_x = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 1].reshape(-1, 1) + (x[:, 0]*x[:, 2]).reshape(-1, 1) #+ x[:, 4].reshape(-1, 1) # x # 20240830
            grads = torch.ones(b_z.shape, device=self.device)
            grad_x = grad(b_z, x, create_graph=True, grad_outputs=grads)[0]
            c_z = (grad_x[:, 0]).reshape(-1, 1) # v_z
            grads = torch.ones(b_x.shape, device=self.device)
            grad_x = grad(b_x, x, create_graph=True, grad_outputs=grads)[0]
            c_x = (grad_x[:, 0]).reshape(-1, 1) # v_x
            return torch.cat([b_z, c_z, b_x, c_x], dim=1) # Throwing
        elif physics == 'Swinging_lin': # PI-MBPO, PINN-MCTS without PINNSim
            b = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + x[:, 1].reshape(-1, 1) # theta # 20240830
            grads = torch.ones(b.shape, device=self.device)
            grad_x = grad(b, x, create_graph=True, grad_outputs=grads)[0]
            c = (grad_x[:, 0]).reshape(-1, 1) # omega
            return torch.cat([b, c], dim=1) # Swinging
        elif physics == 'Sliding': # PI-MBPO, PINN-MCTS without PINNSim
            b = (torch.log(torch.ones(x[:, 2].reshape(-1, 1).shape, device = self.device) + x[:, 2].reshape(-1, 1))) * torch.sin(x[:, 2].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + (x[:, 2]*x[:, 1]).reshape(-1, 1) # S # 20240830
            grads = torch.ones(b.shape, device=self.device)
            grad_x = grad(b, x, create_graph=True, grad_outputs=grads)[0]
            c = (grad_x[:, 2]).reshape(-1, 1) # v
            return torch.cat([c, b], dim=1) # Sliding
        elif physics == 'Throwing': # PI-MBPO, PINN-MCTS without PINNSim
            b_z = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 0].reshape(-1, 1) + (x[:, 0]*x[:, 1]).reshape(-1, 1) + x[:, 3].reshape(-1, 1) # z # 20240830
            b_x = (torch.log(torch.ones(x[:, 0].reshape(-1, 1).shape, device = self.device) + x[:, 0].reshape(-1, 1))) * torch.sin(x[:, 0].reshape(-1, 1)) * a[:, 1].reshape(-1, 1) + (x[:, 0]*x[:, 2]).reshape(-1, 1) + x[:, 4].reshape(-1, 1) # x # 20240830
            grads = torch.ones(b_z.shape, device=self.device)
            grad_x = grad(b_z, x, create_graph=True, grad_outputs=grads)[0]
            c_z = (grad_x[:, 0]).reshape(-1, 1) # v_z
            grads = torch.ones(b_x.shape, device=self.device)
            grad_x = grad(b_x, x, create_graph=True, grad_outputs=grads)[0]
            c_x = (grad_x[:, 0]).reshape(-1, 1) # v_x
            return torch.cat([b_z, c_z, b_x, c_x], dim=1) # Throwing
        else:
            return a