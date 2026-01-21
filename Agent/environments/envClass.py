from isaacgym import gymapi, gymutil
import os
import cv2
import math
import torch
import numpy as np
from math import sqrt
from urdfpy import URDF
from detection import Detector
from abc import ABC, abstractmethod


class Environment(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def setup(self, gym, sim, env, viewer, args):
        pass

    @abstractmethod
    def setup(self, gym, sim, env, viewer, args):
        pass

    @abstractmethod
    def generate_specific_state(self, gym, sim, env, viewer, args, config, goal_center):
        pass

    @abstractmethod
    def generate_random_state(self, gym, sim, env, viewer, args, config):        
        pass

    @abstractmethod
    def get_goal_height(self, gym, sim, env, viewer, args, config):    
        pass

    @abstractmethod
    def get_piece_height(self, gym, sim, env, viewer, args, config):    
        pass

    @abstractmethod
    def get_goal_position_from_simulator(self, gym, sim, env, viewer, args, config):    
        pass

    @abstractmethod
    def get_piece_position_from_simulator(self, gym, sim, env, viewer, args, config):    
        pass

    @abstractmethod
    def get_goal_position(self, gym, sim, env, viewer, args, config):    
        pass

    @abstractmethod
    def get_piece_position(self, gym, sim, env, viewer, args, config):    
        pass

    @abstractmethod
    def generate_random_action(self):    
        pass

    @abstractmethod
    def execute_action(self, gym, sim, env, viewer, args, config, action, return_ball_pos=False):    
        pass

    @abstractmethod
    def execute_action_pinn(self, config, goal_pos, action):    
        pass

    @abstractmethod
    def execute_random_action(self, gym, sim, env, viewer, args, config, return_ball_pos=False):    
        pass
