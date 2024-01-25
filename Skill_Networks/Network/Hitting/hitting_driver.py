import torch
import torch.autograd as autograd
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
from hitting_ff import ff_driver

config_filename = "hitting_config.yaml"

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

def train():
    ff_driver(config)

train()

