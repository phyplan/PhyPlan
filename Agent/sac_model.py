import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

NUM_COLORS = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureEnv(gym.Env):
    def __init__(self, feature=None, reward = None, action_shape=1, feature_shape=512):
        super(FeatureEnv, self).__init__()
        self.feature = feature
        self.reward = reward
        self.current_index = 0
        self.action_space = spaces.Box(low=-10, high=10, shape=(action_shape,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_shape,), dtype=np.float32)

    def step(self, output):
        reward = -100*np.square(self.reward[self.current_index] - output) 
        info = {}
        self.current_index += 1
        if self.current_index >= len(self.feature):
            done = True
            self.current_index = 0
        else:
            done = False
        truncated = done
        observation = self.feature[self.current_index].T

        return observation, reward, done, truncated, info

    def reset(self,seed=None, options=None):
        self.current_index = 0
        info = {} 
        return self.feature[self.current_index].T , info 
          
    def render(self, mode='human'):
        pass 


class ActionNetwork(nn.Module):
    
    def __init__(self, action_size, output_size, hidden_size=256, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(action_size, hidden_size)])
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, tensor):
        for layer in self.layers:
            tensor = nn.functional.relu(layer(tensor), inplace=True)
        return self.output(tensor)


class FilmActionNetwork(nn.Module):

    def __init__(self, action_size, output_size, **kwargs):
        super().__init__()
        self.net = ActionNetwork(action_size, output_size * 2, **kwargs)

    def forward(self, actions, image):
        beta, gamma = torch.chunk(self.net(actions).unsqueeze(-1).unsqueeze(-1), chunks=2, dim=1)
        return image * beta + gamma

class ResNet18FilmAction(nn.Module):

    def __init__(self,
                 action_size,
                 action_layers=1,
                 action_hidden_size=256,
                 fusion_place='last'):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=False)
        conv1 = nn.Conv2d(NUM_COLORS,
                          64,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False)
        self.register_buffer('embed_weights', torch.eye(NUM_COLORS))
        # self.embed_weights = nn.Parameter(torch.eye(NUM_COLORS))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList(
            [net.layer1, net.layer2, net.layer3, net.layer4])

        def build_film(output_size):
            return FilmActionNetwork(action_size,
                                     output_size,
                                     hidden_size=action_hidden_size,
                                     num_layers=action_layers)

        assert fusion_place in ('first', 'last', 'all', 'none', 'last_single')

        self.last_network = None
        if fusion_place == 'all':
            self.action_networks = nn.ModuleList(
                [build_film(size) for size in (64, 64, 128, 256)])
        elif fusion_place == 'last':
            # Save module as attribute.
            self._action_network = build_film(256)
            self.action_networks = [None, None, None, self._action_network]
        elif fusion_place == 'first':
            # Save module as attribute.
            self._action_network = build_film(64)
            self.action_networks = [self._action_network, None, None, None]
        elif fusion_place == 'last_single':
            # Save module as attribute.
            self.last_network = build_film(512)
            self.action_networks = [None, None, None, None]
        elif fusion_place == 'none':
            self.action_networks = [None, None, None, None]
        else:
            raise Exception('Unknown fusion place: %s' % fusion_place)
        

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'

    def preprocess(self, observations):
        # image = self._image_colors_to_onehot(observations)
        image = observations.to(device)
        features = self.stem(image)
        for stage, act_layer in zip(self.stages, self.action_networks):
            if act_layer is not None:
                break
            features = stage(features)
        else:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        return dict(features=features)

    def forward(self, observations, actions, preprocessed=None):
        if preprocessed is None:
            preprocessed = self.preprocess(observations)
        return self._forward(actions, **preprocessed)

    def _forward(self, actions, features):
        actions = actions.to(features.device)
        skip_compute = True
        for stage, film_layer in zip(self.stages, self.action_networks):
            if film_layer is not None:
                skip_compute = False
                features = film_layer(actions, features)
            if skip_compute:
                continue
            features = stage(features)
        if not skip_compute:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        if self.last_network is not None:
            features = self.last_network(actions, features)
        features = features.flatten(1)
        if features.shape[0] == 1 and actions.shape[0] != 1:
            # Haven't had a chance to use actions. So will match batch size as
            # in actions manually.
            features = features.expand(actions.shape[0], -1)
        return features

    def ce_loss(self, decisions, targets):
        targets = targets.to(dtype=torch.float, device=decisions.device)
        return nn.functional.binary_cross_entropy_with_logits(
            decisions, targets)

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices.to(dtype=torch.long, device=device),
            self.embed_weights)
        onehot = onehot.permute(0, 3, 1, 2).contiguous()
        return onehot

class ImageDataset(Dataset):
    def __init__(self, image_folder, data_path, num_contexts):
        self.image_folder = image_folder
        self.num_contexts = num_contexts
        f = open(data_path, 'r')
        self.actions = {}
        self.rewards = {}
        for line in f.readlines():
            line = [float(_) for _ in line.strip().split()]
            line[0] = int(line[0])
            line[1] = int(line[1])
            if line[0] not in self.actions:
                self.actions[line[0]] = []
                self.rewards[line[0]] = []
            self.actions[line[0]].append(line[2:-1])
            self.rewards[line[0]].append(line[-1])

    def __len__(self):
        files = os.listdir(self.image_folder)
        # return len(files) - 1
        return min(len(files) - 1, self.num_contexts)
    
    def __getitem__(self, idx):
        item = {}
        img = read_image(os.path.join(self.image_folder, 'rgb_{}.png'.format(idx))).float()
        item['img'] = img.unsqueeze(0)
        item['action'] = torch.tensor(self.actions[idx])
        item['reward'] = torch.tensor(self.rewards[idx])
        return item