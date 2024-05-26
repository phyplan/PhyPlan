import os
import ast
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
import ast
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

class DebugCallback(BaseCallback):
    def _on_step(self) -> bool:
        print("Step", self.num_timesteps)
        return True




parser = argparse.ArgumentParser(description='Process custom parameters for gym environment setup.')

 
parser.add_argument('--env', type=str, default='pendulum', help='Specify Environment')
parser.add_argument('--fast', type=ast.literal_eval, default=False, help='Invoke Fast Simulator')
parser.add_argument('--epochs', type=int, default=40, help='Number of Epochs')
parser.add_argument('--contexts', type=int, default=100, help='Number of Contexts')
parser.add_argument('--contexts_step', type=int, default=10, help='Number of Step Contexts')
parser.add_argument('--action_params', type=int, default=2, help='Number of Action Parameters')

 
args = parser.parse_args()

print(args.env)
print(args.epochs)
print(args.contexts)
print(args.contexts_step)
print(args.action_params)

 


NUM_EPOCHS = args.epochs
NUM_CONTEXTS = args.contexts
NUM_CONTEXTS_STEP = args.contexts_step
ACTION_PARAMETERS = args.action_params

NUM_CONTEXTS_STEP = 50
NUM_CONTEXTS = 50

train_folder = 'data/images_train_' + args.env + ('_pinn' if args.fast else '_actual')
test_folder = 'data/images_test_' + args.env + ('_pinn' if args.fast else '_actual')
train_file = 'data/actions_train_' + args.env + ('_pinn' if args.fast else '_actual') + '.txt'
test_file = 'data/actions_test_' + args.env + ('_pinn' if args.fast else '_actual') + '.txt'

dataset = ImageDataset(train_folder, train_file, NUM_CONTEXTS)
test_dataset = ImageDataset(test_folder, test_file, NUM_CONTEXTS)
dataloader = DataLoader(dataset, batch_size = 1)
feature_class = ResNet18FilmAction(ACTION_PARAMETERS, fusion_place = 'last_single').to(device)


device_sac = get_device("cuda")  # Can be "auto", "cuda", or "cpu"
env = FeatureEnv()
# model = SAC("MlpPolicy", env, verbose=2)
#model = PPO("MlpPolicy", env, verbose=0)
model = PPO("MlpPolicy", env, learning_rate=1e-4, clip_range=0.2, ent_coef=0.01, verbose=0)
 

def testLoss(model):
    running_loss = 0.0 
    # with torch.no_grad():
    for i in range(len(test_dataset)):
        if i == len(test_dataset):
            break
        item = test_dataset[i]
        img = item['img'].to(device)
        action = item['action'].to(device)
        reward = item['reward'].detach().cpu().numpy()
        feature = feature_class(img, action).detach().cpu().numpy()
        num_actions = len(reward)
        for j in range(0,num_actions):
            feature_test = feature[j].T
            output, _ = model.predict(feature_test, deterministic=True)
            loss = np.abs(output[0] - reward[j])/np.abs(reward[j])
            running_loss += loss
        if i == 0:
            print('OUT:', output)
            print('REW:', reward[i])
    return running_loss / (len(test_dataset)*num_actions)




for num_contexts in range(NUM_CONTEXTS_STEP, NUM_CONTEXTS + 1, NUM_CONTEXTS_STEP):

    running_loss = 0.0
    count = 0
    least_test_loss = 1e9

    for epoch in range(NUM_EPOCHS):
        print(epoch)
        with  tqdm(dataset) as tepoch:
            features = []
            rewards = []
       
            for idx, item in enumerate(tepoch):
               
                if idx > num_contexts or idx == len(dataset) - 1:
                    
                    break
          
                img = item['img'].to(device)
                action = item['action'].to(device)
                reward = item['reward']
                feature = feature_class(img, action).detach().cpu().numpy()
                env.reward = reward
                env.feature = feature
                time_step = len(reward)
                 
                model.learn(total_timesteps=time_step,log_interval=1,
                            # callback=DebugCallback()
                )
                
                # test_loss = testLoss(model)
                # mean_reward, std_dev = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
                # print(f"Mean Reward: {mean_reward}, Std Dev: {std_dev}")
                # print('Test Loss: {}'.format(test_loss))
                
        test_loss = testLoss(model)
        mean_reward, std_dev = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f"Mean Reward: {mean_reward}, Std Dev: {std_dev}")
        print('Test Loss: {}'.format(test_loss))
        running_loss = 0.0
        count = 0

        if test_loss < least_test_loss:
            save_path = 'agent_models/ppo_model_%s_%s_%d.pth' % (args.env, ('pinn' if args.fast else 'actual'), num_contexts)
            model.save(save_path)
            least_test_loss = test_loss
            print('Model Saved!')
