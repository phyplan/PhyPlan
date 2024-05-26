import ast
import time
import requests
from flask import Flask, request, jsonify
import os
import sys
import torch
import torchvision
from sac_model import *
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from stable_baselines3 import SAC, PPO
import gymnasium as gym
from gymnasium import spaces
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import argparse
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


 
env = FeatureEnv()
model = PPO("MlpPolicy", env, learning_rate=1e-4, clip_range=0.2, ent_coef=0.01, verbose=0)


def get():
    response = request.get_json()
    # print('--------------- RESPONSE: --', response)
    
    iter = response['iter']
   
        #loading_model
    env = response['env']
    model = PPO.load(response['model'])
    
    #response to feature
    feature = np.array(response['feature'])
    reward = model.predict(feature, deterministic=True)[0][0][0]
    #return data
    data = {
            'reward': int(reward*1000000)
        }
    return jsonify(data)

 


app = Flask(__name__)
app.route('/send', methods=['POST'])(get)
app.run(host='0.0.0.0', port=7001)




 
 



