import torch
import torch.autograd as autograd
import torch
import numpy as np
import pandas as pd
import os
import sys
from model import train_model, evaluate_model

import argparse

parser = argparse.ArgumentParser(description="Skill Learning Model Configuration")

parser.add_argument('--batch_size', type=int, default=0, help='Batch size for training')
parser.add_argument('--early_stopping', type=int, default=0, help='Early stopping criteria')

parser.add_argument('--evaluate', type=bool, default=False, help='Set to True to evaluate, False to train')
parser.add_argument('--load_only', type=bool, default=False, help='True to only load model and evaluate')
parser.add_argument('--n_models', type=int, default=0, help='Number of models to train and take mean over')

parser.add_argument('--save_eval', type=bool, default=False, help='Save evaluated data')
parser.add_argument('--save_plot', type=bool, default=False, help='Save evaluation plot')
parser.add_argument('--save_model', type=bool, default=False, help='Save trained model')

parser.add_argument('--model_dir', type=str, default="", help='Model save directory')
parser.add_argument('--model_name', type=str, default="", help='Name of the model')
parser.add_argument('--load_dir', type=str, default="", help='Directory to load the model')
parser.add_argument('--eval_out', type=str, default="", help='CSV file to save evaluation results')

parser.add_argument('--cuda_enabled', type=bool, default=False, help='Enable CUDA for GPU computation')
parser.add_argument('--num_data', type=int, default=0, help='Number of training data points')
parser.add_argument('--num_coll', type=int, default=0, help='Number of collocation data points')

parser.add_argument('--skill', type=str, default="", help='Skill type (Sliding, Bouncing, Hitting, Swinging, Throwing)')
parser.add_argument('--parameter', type=str, default="", help='Skill parameter type (e.g., for Swinging: sin, lin, lns)')

parser.add_argument('--dataset', type=str, default="", help='Path to training dataset')
parser.add_argument('--data_val', type=str, default="", help='Path to validation dataset')
parser.add_argument('--coll', type=str, default="", help='Path to collision dataset')

parser.add_argument('--train_mode', type=str, default="", help='Training mode (nn or pinn)')
parser.add_argument('--physics_only', type=bool, default=False, help='True if only physics-based learning is used')

parser.add_argument('--num_layers', type=int, default=0, help='Number of layers in the model')
parser.add_argument('--neurons_per_layer', type=int, default=0, help='Number of neurons per layer in the model')
parser.add_argument('--alpha', type=float, default=0.0, help='Learning rate or regularization parameter')

args = parser.parse_args()

args_dict = vars(args)
lowercase_keys = {'num_layers', 'neurons_per_layer', 'alpha'}
config = {k if k in lowercase_keys else k.upper(): v for k, v in args_dict.items()}

def start():
    if config['SKILL'] == 'Hitting':
        values = [1,4,4,5]

    elif config['SKILL'] =='Bouncing':
        values = [3,7,7,9]

    elif config['SKILL'] =='Sliding':
        if config['PARAMETER'] == 'Fixed' or config['PARAMETER'] == 'Determine':
            values = [3,5,5,7]
        else:
            values = [2,5,5,7]

    elif config['SKILL'] =='Throwing':
        values = [1,6,6,10]

    elif config['SKILL'] =='Swinging':
        values = [1,3,3,5]
    
    elif config['SKILL'] == 'Collision':
        if config['PARAMETER'] == 'Fixed' or config['PARAMETER'] == 'Determine':
            values = [2,4,4,6]
        else:
            values = [1,4,4,6]

    else:
        print("Skill not selected!")
        return
    if config['EVALUATE']:
        if not(config["LOAD_ONLY"]):
            a = [None]*config["N_MODELS"]
            if  (config["SKILL"] == 'Sliding' or config['SKILL'] == 'Collision') and config["PARAMETER"] == 'Determine':
                mu = np.zeros(config["N_MODELS"])
            for i in range(config["N_MODELS"]):
                print("Training model {}".format(i + 1))
                train_model(config, values, i)
                print("Evaluating model {}".format(i + 1))
                if (config["SKILL"] == 'Sliding' or config['SKILL'] == 'Collision') and config["PARAMETER"] == 'Determine':
                    a[i], mu[i] = evaluate_model(config, values, i)
                else:
                    a[i] = evaluate_model(config, values, i)
            a = np.array(a)
            a.sort(axis=0)
            print("Mean is {}".format(a.mean(axis=0)))
            print("STDEV is {}".format(a.std(axis=0)))
            print("Losses are {}".format(np.array2string(a, separator='\n', max_line_width=np.inf)))
            if  (config["SKILL"] == 'Sliding' or config['SKILL'] == 'Collision') and config["PARAMETER"] == 'Determine':
                mu.sort()
                print("Learnt prm are {}".format(np.array2string(mu, separator='\n', max_line_width=np.inf)))
        else:
            evaluate_model(config, values)
    else:
        train_model(config,values)
    return


if __name__ == '__main__':
    start()

