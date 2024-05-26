import torch
import torch.autograd as autograd
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
from model import train_model


config_filename = "config.yaml"
with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

def train():
    if config['SKILL'] == 'Hitting':
        values = [1,4,4,5]

    elif config['SKILL'] =='Bouncing':
        values = [3,7,7,9]

    elif config['SKILL'] =='Sliding':
        if config['PARAMETER'] == 'Fixed':
            values = [3,5,5,7]
        elif config['PARAMETER'] == 'Determine':
            values = [2,5,5,7]
        else:
            values = [2,5,5,7]

    elif config['SKILL'] =='Throwing':
        values = [2,5,5,8]

    elif config['SKILL'] =='Swinging':
        values = [2,4,4,6]

    else:
        print("Skill not selected!")
        return

    train_model(config,values)
    return


if __name__ == '__main__':
    train()

