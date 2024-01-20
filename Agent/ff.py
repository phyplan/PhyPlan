from numpy.lib.npyio import save
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
VT_test = None
X_test = None
VT_validation = None
X_validation = None
VT_train = None
X_train = None

optimizer = None

PRINT_ON_ITER = 100

def prepare_data(config):
    data_df = pd.read_csv(config['model_name'] + "data.csv", header=None)
    data = np.array(data_df)
    print(data.shape)
    req_data = []
    # for d in data:
    #     if d[0] <= config['num_sim']:
    #         req_data.append(d[2:])
    req_data = data[:,1:]
    req_data = np.array(req_data)
    np.random.shuffle(req_data)
    return req_data[:,:2], req_data[:,2:]
    # if config['model_name'] == 'pendulum':
    #     return req_data[:,:2] , req_data[:,2:]
    # elif config['model_name'] == 'var_sliding':
    #     return req_data[:,:3] , req_data[:,3:]
    # elif config['model_name'] == 'same_sliding':
    #     return req_data[:,1:3] , req_data[:,3:]
    # else:
    #     return req_data[:,:3] , req_data[:,3:]

def make_data_noisy(data):
    noisy_data =np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            noisy_data[i,j] = (data[i,j] * random.uniform(0.95, 1.05))
    return noisy_data


def set_loss(model, device, batch_size, data = None, X_true = None):
	error = 0
	with torch.no_grad():
		predicted = model.forward(data)
	error = (predicted - X_true)
	loss_u = torch.sum(error ** 2) / (error.shape[0])
	return loss_u

def get_result(model, device, batch_size, data = None, X_true = None):
	with torch.no_grad():
		predicted = model.forward(data)
	return predicted


class FF_Baseline(nn.Module):

    def __init__(self, VT_u, X_u, layers, lb, ub, device, config, N_u):
        super().__init__()

        self.id = id
        self.device = device

        self.u_b = ub
        self.l_b = lb
        self.N_u = N_u
        self.batch_size = config['BATCH_SIZE']
        self.config = config

        self.VT_u = VT_u
        self.X_u = X_u
        self.layers = layers

        self.activation = nn.ReLU()
        # self.loss_function = nn.MSELoss(reduction ='mean') # removing for being able to batch
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0
        self.elapsed = None

        self.iter_history = []
        self.history = None # train, validation

        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).to(self.device)

        # Preprocessing input - scaled from 0 to 1
        a = x.float()

        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)

        # Activation is not applied to last layer
        a = self.linears[-1](a)
        return a

    def loss(self, x, y):
        prediction = self.forward(x)
        error = prediction - y
        loss_u = torch.sum(error ** 2) / (error.shape[0])
        return loss_u

    def closure(self):
        """ Called multiple times by optimizers like Conjugate Gradient and LBFGS.
        Clears gradients, compute and return the loss.
        """
        optimizer.zero_grad()
        loss = self.loss(self.VT_u, self.X_u)
        loss.backward()        # To get gradients

        self.iter += 1

        if self.iter % PRINT_ON_ITER == 0:
            training_loss = loss.item()
            validation_loss = set_loss(self, self.device, self.batch_size, VT_validation, X_validation).item()
            print(
                'Iter %d, Training: %.5e, Validation: %.5e' % (self.iter, training_loss, validation_loss)
            )
            self.iter_history.append(self.iter)
            current_history = np.array([training_loss, validation_loss])
            if self.history is None: self.history = current_history
            else: self.history = np.vstack([self.history, current_history])

        return loss

    def plot_history(self, debug=False):
        """ Saves training (loss_u + loss_f and both separately) and validation losses
        """
        loss = {}
        if self.history is not None:
            epochs = self.iter_history
            loss['Training'] = np.ndarray.tolist(self.history[:,0].ravel())
            loss['Validation'] = np.ndarray.tolist(self.history[:,1].ravel())
        else:
            epochs = [self.iter]
            loss['Training'] = [1e6]
            loss['Validation'] = [1e6]
        last_training_loss = loss['Training'][-1]
        last_validation_loss = loss['Validation'][-1]

        for loss_type in loss.keys():
            plt.clf()
            plt.plot(epochs, loss[loss_type], color = (63/255, 97/255, 143/255), label=f'{loss_type} loss')
            if (loss_type == 'Validation'): title = f'{loss_type} loss (Relative MSE)\n'
            else : title = f'{loss_type} loss (MSE)\n'

            plt.title(
                title +
                f'Elapsed: {self.elapsed:.2f}, N_u: {self.N_u},\n Validation: {last_validation_loss:.2f}, Train: {last_training_loss:.2f}'
            )
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            savefile_name = ''
            if debug: savefile_name += 'Debug_'
            savefile_name += 'plot_' + self.config['model_name']
            # if debug: savefile_name += '_' + str(self.N_f) + '_' + str(self.alpha)
            savefile_name += '_' + loss_type
            savefile_name += '.png'
            savedir = self.config['modeldir']
            if debug: savedir += self.config['model_name'] + '/'

            print(savedir , savefile_name)
            if self.config['SAVE_PLOT']: plt.savefig(savedir + savefile_name)
            plt.close()


def pendulum_driver(config):
    plt.figure(figsize=(8, 6), dpi=80)
    num_layers = config['num_layers'] #8
    num_neurons = config['neurons_per_layer'] #40

    torch.set_default_dtype(torch.float)
    torch.manual_seed(config['seed']) #rnd val
    np.random.seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() and config['CUDA_ENABLED'] else 'cpu')

    print("Running this on", device)
    if device == 'cuda':
        print(torch.cuda.get_device_name())

    global VT_train, X_train, VT_validation, X_validation

    VT_train, X_train = prepare_data(config)
    if config['Noise']:
        VT_train = make_data_noisy(VT_train)
        X_train = make_data_noisy(X_train)
    VT_train = torch.from_numpy(VT_train).float().to(device)
    X_train = torch.from_numpy(X_train).float().to(device)
    VT_validation, X_validation = VT_train, X_train
    print(VT_train.shape, X_train.shape)
    # layers is a list, not an ndarray
    layers = np.concatenate([[VT_train.shape[1]], num_neurons*np.ones(num_layers), [X_train.shape[1]]]).astype(int).tolist()

    models = []
    validation_losses = []

    N_u = config['num_sim'] # datapoints

    lb = 5.0000
    ub = 15.00

    print(f'++++++++++ N_u:{N_u} ++++++++++')

    model = FF_Baseline(VT_train, X_train, layers, lb, ub, device, config, N_u)
    model.to(device)
    # print(model)

    # L-BFGS Optimizer
    global optimizer
    # optimizer = torch.optim.LBFGS(
    #     model.parameters(), lr=0.01,
    #     max_iter = config['EARLY_STOPPING'],
    #     tolerance_grad = 1.0 * np.finfo(float).eps,
    #     tolerance_change = 1.0 * np.finfo(float).eps,
    #     history_size = 100
    # )
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=0.01
    # )
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.01,
        max_iter = config['EARLY_STOPPING'],
        tolerance_grad = 1.0 * np.finfo(float).eps,
        tolerance_change = 1.0 * np.finfo(float).eps,
        history_size = 100
    )
    # prev_loss = -np.inf
    # current_loss = np.inf

    start_time = time.time()

    optimizer.step(model.closure)        # Does not need any loop like Adam
    # pbar = tqdm(total=config['EARLY_STOPPING'])
    # while abs(current_loss-prev_loss)>np.finfo(float).eps and model.iter<config['EARLY_STOPPING']:
    #     pbar.update(1)
    #     current_loss, prev_loss = model.closure(), current_loss
    #     optimizer.step()

    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    validation_loss = set_loss(model, device, config['BATCH_SIZE'], VT_validation, X_validation)
    model.elapsed = elapsed
    model.plot_history()
    model.to('cpu')
    models.append(model)
    validation_losses.append(validation_loss.cpu().item())

    model_id = np.nanargmin(validation_losses) # choosing best model out of the bunch
    model = models[model_id]

    """ Model Accuracy """
    error_validation = validation_losses[model_id]
    print('Validation Error of finally selected model: %.5f'  % (error_validation))

    """" For plotting final model train and validation errors """
    if config['SAVE_PLOT']: model.plot_history(debug=False)

    """ Saving only final model for reloading later """
    if config['SAVE_MODEL']: torch.save(model, config['modeldir'] + config['model_name'] + '_' + str(config['num_sim']) + '_' + str(config['Noise']) + '.pt')

    # all_hyperparameter_models = [[models[md].N_u, validation_losses[md]] for md in range(len(models))]
    # all_hyperparameter_models = pd.DataFrame(all_hyperparameter_models)
    # all_hyperparameter_models.to_csv(config['modeldir'] + config['model_name'] + '.csv', header=['N_u', 'Validation Error'])

    if device == 'cuda':
        torch.cuda.empty_cache()

# if __name__ == "__main__":
#     main_loop(config['num_datadriven'], config['num_collocation'], config['num_layers'], config['neurons_per_layer'], config['num_validation'])
