import torch
import torch.autograd as autograd
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
from var_sliding_ff import ff_driver
from var_sliding_pidnn import pidnn_driver

config_filename = "var_sliding_config.yaml"

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

def train():
    if config['TRAIN_MODE'] == 'pidnn':
        pidnn_driver(config)
    else:
        ff_driver(config)

train()

