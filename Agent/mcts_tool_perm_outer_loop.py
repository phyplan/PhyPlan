from isaacgym import gymapi, gymutil
import ast
import copy
import math
import time
import torch
import torch.nn as nn
import multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mcts_agent_softgreedy import MCTS_Agent
from collections import deque

# mp.set_start_method('spawn')

args = gymutil.parse_arguments(custom_parameters=[
    {
        "name": "--vis",
        "type": ast.literal_eval,
        "default": False,
        "help": "Render a Visualization of Tool Selection"
    },
    {
        "name": "--contexts",
        "type": int,
        "default": 1,
        "help": "Number of Contexts"
    },
    {
        "name": "--actions",
        "type": int,
        "default": 1,
        "help": "Number of Actions"
    },
    {
        "name": "--D",
        "type": int,
        "default": 20,
        "help": "Discretisation Parameter"
    },
    {
        "name": "--env",
        "type": str,
        "default": "pendulum",
        "help": "Environment"
    },
    {
        "name": "--simulate",
        "type": ast.literal_eval,
        "default": False,
        "help": "Render Simulation"
    },
    {
        "name": "--robot",
        "type": ast.literal_eval,
        "default": False,
        "help": "Make Robot perform the actions"
    },
    {
        "name": "--perception",
        "type": ast.literal_eval,
        "default": False,
        "help": "Use Perception"
    },
    {
        "name": "--pinn_attempts",
        "type": int,
        "default": 20,
        "help": "Number of PINN Attempts"
    },
    {
        "name": "--adaptive",
        "type": ast.literal_eval,
        "default": True,
        "help": "Use GP-UCB"
    },
    {
        "name": "--widening",
        "type": ast.literal_eval,
        "default": False,
        "help": "Use Progressive Widening"
    },
])

args.fast = False

np.random.seed(0)

if args.env == 'pendulum':
    from environments.pendulum_with_perm import Env
elif args.env == 'sliding':
    from environments.sliding_with_perm import Env
elif args.env == 'wedge':
    from environments.wedge_with_perm import Env
elif args.env == 'sliding_bridge':
    from environments.sliding_bridge_with_perm import Env
elif args.env == 'pendulum_wedge':
    from environments.pendulum_wedge_with_perm import Env

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NUM_CONTEXTS = args.contexts
NUM_ACTIONS = args.actions
NUM_SPLITS = args.D

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1 / 200

sim_params.substeps = 4
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
# sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
plane_params.static_friction = 0
plane_params.dynamic_friction = 0
plane_params.restitution = 1
gym.add_ground(sim, plane_params)

# Comment for realistic lighting
# gym.set_light_parameters(sim, 0, gymapi.Vec3(1, 1, 1), gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, -1))

viewer = None
if args.simulate:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

env_spacing = 20
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

if args.simulate:
    main_cam_pos = gymapi.Vec3(-1, -1, 2)
    main_cam_target = gymapi.Vec3(0.2, -0.2, 0.0)

    if args.env == 'sliding_bridge':
        main_cam_pos = gymapi.Vec3(-1, 1, 2)
        main_cam_target = gymapi.Vec3(0.2, -0.2, 0.0)
    if args.env == 'wedge':
        main_cam_pos = gymapi.Vec3(-1, -1, 2)
        main_cam_target = gymapi.Vec3(0.0, -0.0, 0.5)
    if args.env == 'pendulum':
        main_cam_pos = gymapi.Vec3(-1, 1, 2)
        main_cam_target = gymapi.Vec3(0.2, -0.2, 0.0)
    if args.env == 'pendulum_wedge':
        main_cam_pos = gymapi.Vec3(-1, 1, 2)
        main_cam_target = gymapi.Vec3(0.2, -0.2, 0.0)
    if args.env == 'sliding':
        main_cam_pos = gymapi.Vec3(1.5, -1.5, 2)
        main_cam_target = gymapi.Vec3(-0.5, -0.5, 0.0)

    gym.viewer_camera_look_at(viewer, None, main_cam_pos, main_cam_target)


task_env = Env()
config = task_env.setup(gym, sim, env, viewer, args)
# if args.env == "pendulum":
#     config = task_env.setup(gym, sim, env, viewer, args, ball_density=30000)
# elif args.env == "wedge":
#     config = task_env.setup(gym, sim, env, viewer, args)
# elif args.env == "sliding":
#     config = task_env.setup(gym, sim, env, viewer, args, use_heavy_puck=True)
# elif args.env == "sliding_bridge":
#     config = task_env.setup(gym, sim, env, viewer, args, use_heavy_puck=True)
# elif args.env == "pendulum_wedge":
#     config = task_env.setup(gym, sim, env, viewer, args, ball_density=30000)

agent = MCTS_Agent(gym, sim, env, viewer, args, task_env, config, task_env.bnds)

def eval_reward(goal_pos, tools, args, execute_action=False):
    print("Tools:", tools)
    try:
        local_config = task_env.setup_tool(gym, sim, env, config, tools)
        agent.updateConfig(local_config)
        agent.updateBounds(task_env.bnds)
        agent.updateArgs(args)
        reward, opt, regret = agent.search(goal_pos, execute_action)
    except Exception as e:
        print(e)
        reward, regret = 0.0, 1.0
    print("Regret:", regret)
    return reward, regret


class Node():
    def __init__(self, depth=0, parent=None, action_chain=[]):
        self.parent = parent
        self.depth = depth
        self.children = []
        self.actions = []
        self.num_actions = []
        self.val_children = []
        self.max_val = []
        self.action_chain = action_chain

    def copy(self):
        newNode = Node(depth=self.depth)
        if self.parent is not None:
            newNode.parent = self.parent.copy()
        newNode.children = self.children.copy()
        newNode.actions = self.actions.copy()
        newNode.num_actions = self.num_actions.copy()
        newNode.val_children = self.val_children.copy()
        newNode.max_val = self.max_val.copy()
        newNode.action_chain = self.action_chain.copy()
        return newNode


class Tool_Selector_MCTS():
    def __init__(self, options, args, execute_action):
        self.ACTION_PARAMETERS = len(options)
        self.ATTEMPTS = 2
        self.C = 1.0
        self.tools = options
        self.args = args
        self.execute_action = execute_action
        self.state_dict = {}
        self.val_dict = {}
        self.variable_list = []
        self.best_regrets = []

    def rollout(self, node, goal_pos):
        action_chain = node.action_chain.copy()
        for i in range(node.depth, self.ACTION_PARAMETERS):
            action_chain.append(np.random.choice(self.tools[i]))
        #self.state_dict[tuple(action_chain)] += 1
        reward, regret = eval_reward(goal_pos, action_chain, self.args, self.execute_action)
        # reward, regret = call_eval_proc(goal_pos, action_chain, self.args)
        return reward

    def kickstart(self, node, goal_pos):
        node.actions = self.tools[node.depth]
        for action in node.actions:
            node.num_actions.append(1)
            child_node = Node(node.depth + 1, node, node.action_chain.copy() + [action])
            node.children.append(child_node)
            reward = self.rollout(child_node, goal_pos)
            node.val_children.append(reward)
            node.max_val.append(reward)

    def MCTS(self, node, goal_pos):
        if node.depth == self.ACTION_PARAMETERS:
            return self.rollout(node, goal_pos)
        max_val = -1e9
        max_idx = None
        A = len(node.children)
        if A == 0:
            self.kickstart(node, goal_pos)
            reward = self.rollout(node, goal_pos)
        else:
            for i in range(A):
                curr_val = node.val_children[i] + self.C * math.sqrt(math.log(sum(node.num_actions)) / node.num_actions[i])
                if curr_val > max_val:
                    max_val = curr_val
                    max_idx = i
            child_node = node.children[max_idx]
            reward = self.MCTS(child_node, goal_pos)
            node.val_children[max_idx] = (
                node.val_children[max_idx] * node.num_actions[max_idx] +
                reward) / (node.num_actions[max_idx] + 1)
            node.num_actions[max_idx] += 1
            node.max_val[max_idx] = max(node.max_val[max_idx], reward)
        return reward

    def get_best_action(self, node, with_reward=False):
        if node.depth == self.ACTION_PARAMETERS or len(node.children) == 0:
            return node.action_chain.copy()
        A = len(node.children)
        max_vis = -1e9
        max_val = -1e9
        max_idx = None
        for i in range(A):
            if with_reward:
                if node.max_val[i] > max_val or (node.max_val[i] == max_val and node.num_actions[i] > max_vis):
                    max_vis = node.num_actions[i]
                    max_val = node.max_val[i]
                    max_idx = i
            else:
                if node.num_actions[i] > max_vis or (node.num_actions[i] == max_vis and node.val_children[i] > max_val):
                    max_vis = node.num_actions[i]
                    max_val = node.val_children[i]
                    max_idx = i
        return self.get_best_action(node.children[max_idx], with_reward)

    def val_store(self, node):
        queue = deque([node]) 
        while queue:
            node = queue.popleft() 
            for child, val in zip(node.children, node.val_children):
                self.val_dict[tuple(child.action_chain)] = val 
                queue.append(child)

    def search(self, goal_pos):
        max_reward = -math.inf
        best_action = None

        root_node = Node()
        prev_conf = -200
        for _ in range(self.ATTEMPTS):
            self.MCTS(root_node, goal_pos)
            action = self.get_best_action(root_node, with_reward=True)
            curr_size = len(action)
            for i in range(curr_size, self.ACTION_PARAMETERS):
                action.append(np.random.choice(self.tools[i]))
            reward, regret = eval_reward(goal_pos, action, self.args, self.execute_action)
            # reward, regret = call_eval_proc(goal_pos, action, self.args)
            if reward > max_reward:
                max_reward = reward
                best_action = action

            max_confidence = 0
            action_node = root_node.copy()
            ready = True
            for i in range(self.ACTION_PARAMETERS):
                num_children = len(action_node.children)
                if num_children == 0:
                    ready = False
                    break
                child_node = action_node.children[0]
                max_conf = 0
                for child in range(num_children):
                    if action_node.max_val[child] > max_conf:
                        max_conf = action_node.max_val[child]
                        child_node = action_node.children[child]
                action_node = child_node.copy()
                max_confidence += max_conf
            max_confidence /= self.ACTION_PARAMETERS

            # if len(self.variable_list) == 0:
            #     self.variable_list.append([])
            #     self.variable_list.append([])
            #     self.variable_list.append([])
            # self.variable_list[0].append([max_confidence, max_reward])

            if not ready:
                continue

            if abs(max_confidence - prev_conf) < 1.0e-4:
                break

            prev_conf = max_confidence

        self.val_store(root_node)
        return best_action, max_reward

    def dict_ret(self):
        return (self.state_dict,self.val_dict)

args_copy = copy.deepcopy(args)
args_copy.actions = 1
args_copy.adaptive = False
args_copy.simulate = False

selector = Tool_Selector_MCTS(options=task_env.available_tools, args=args_copy, execute_action=False)

start = time.time()
img_dir = config['img_dir']
regrets = []
action_wise_regrets = [[] for _ in range(NUM_ACTIONS)]
# vals = []
for iter in range(NUM_CONTEXTS):
    print(f"Context {iter+1}")
    task_env.generate_random_state(gym, sim, env, viewer, args, config)

    if args.perception:
        goal_pos = task_env.get_goal_position(gym, sim, env, viewer, args, config)
        print("GOAL: ", goal_pos)
        print("GOAL_ACT: ", task_env.get_goal_position_from_simulator(gym, sim, env, viewer, args, config))
    else:
        goal_pos = task_env.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

    with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
        pos_f.write(f'Context {iter}\n')
        pos_f.write(f'Goal_Pos: {goal_pos}\n')

    tools, reward = selector.search(goal_pos)
    # state_dict,val_dict = selector.dict_ret()
    # print(state_dict)
    # print(val_dict)
    # vals.append(val_dict.copy())
    agent.best_regrets = []
    agent.args.img_file = "%s/rgb_%d.png" % (img_dir, iter)
    reward, regret = eval_reward(goal_pos, tools, args, execute_action=True)
    # reward, regret = eval_reward(goal_pos, tools, args, execute_action=False)
    # reward, regret = call_eval_proc(goal_pos, tools, args)
    # result = call_eval_proc(goal_pos, tools, args, get_intermediate_regret=True, execute_action=True)
    # reward, regret = result[0], result[1]
    print("**********************************************")
    print("Tools:", tools)
    print("Tool Reward:", reward)
    print("Final Regret:", regret)
    print("**********************************************")
    regrets.append(regret)

    for act in range(NUM_ACTIONS):
        action_wise_regrets[act].append(agent.best_regrets[act])

# print(f'Final:', np.mean(regrets), np.std(regrets))
end = time.time()

f = open('experiments/regret_result_mcts_tool_select_' + args.env + '.txt', 'a')
for i in range(NUM_ACTIONS):
    f.write("MCTS_Outer_Loop_Selected_Tool_Perm\n")
    f.write("Contexts: %f, Actions: %f, D: %f\n" % (NUM_CONTEXTS, i+1, NUM_SPLITS))

    print(f'Final {i}:', np.mean(action_wise_regrets[i]), np.std(action_wise_regrets[i]))
    f.write("Final %f %f\n\n" % (np.mean(action_wise_regrets[i]), np.std(action_wise_regrets[i])))

f.write(f"TIME per context = {(end - start) / NUM_CONTEXTS}\n")
# for val_d in vals:
#     f.write(str(val_d))
f.close()
