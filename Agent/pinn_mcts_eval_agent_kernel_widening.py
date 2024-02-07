from isaacgym import gymapi, gymutil
import ast
import math
import torch
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(custom_parameters=[
    {
        "name": "--action_params",
        "type": int,
        "default": 2,
        "help": "Number of Action Parameters"
    },
    {
        "name": "--contexts",
        "type": int,
        "default": 20,
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
        "default": 10,
        "help": "Number of PINN Attempts"
    },
    {
        "name": "--adaptive",
        "type": ast.literal_eval,
        "default": True,
        "help": "Use GP-UCB"
    },
])

args.fast = False
np.random.seed(0)

if args.env == 'pendulum':
    from environments.pendulum import *
elif args.env == 'sliding':
    from environments.sliding import *
elif args.env == 'wedge':
    from environments.wedge import *
elif args.env == 'sliding_bridge':
    from environments.sliding_bridge import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# ******************** ENVIRONMENT SETUP ******************** #

# ---------- Simulation Setup ----------
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 400
sim_params.substeps = 2
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

# ---------- Uncomment to stop realistic lighting ----------
# gym.set_light_parameters(sim, 0, gymapi.Vec3(1, 1, 1), gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, -1))

viewer = None
if args.simulate:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

env_spacing = 20
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

# ---------- Set camera viewing location ----------
if args.simulate:
    main_cam_pos = gymapi.Vec3(0, 0.001, 3)
    main_cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, main_cam_pos, main_cam_target)


# *********************************************************** #
# ******************** ALGORITHMIC SETUP ******************** #


ACTION_PARAMETERS = args.action_params
PINN_ATTEMPTS = args.pinn_attempts
NUM_CONTEXTS = args.contexts
NUM_ACTIONS = args.actions
NUM_SPLITS = args.D
C = 1.0

kernel = 1 * RBF(length_scale=1.0)
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

config = setup(gym, sim, env, viewer, args)


def evalGP(goalPos, action, actions, rewards, config):
    '''
    Performs the 'action' using PINN models
    
    # Parameters
    goalPos = (x, y) coordinates of the goal, \\
    action  = action to be executed using PINN, \\
    actions = list of actions taken so far, \\
    rewards = rewards corresponding to all previous actions, \\
    config  = configuration of the environment

    # Returns
    reward (adapted using GP-UCB if args.adaptive = True)
    '''
    if len(actions) == 0:
        return execute_action_pinn(config, goalPos, action)
    if args.adaptive:
        mean_pred, std_pred = gaussian_process.predict([action], return_std=True)
        return execute_action_pinn(config, goalPos, action) + mean_pred[0] + 0.5 * std_pred[0]
    else:
        return execute_action_pinn(config, goalPos, action)


class Node():
    def __init__(self, depth=0, parent=None, action_chain=[]):
        self.parent = parent
        self.depth = depth
        self.children = []
        self.actions = []
        self.num_actions = []
        self.val_children = []
        self.action_chain = action_chain


def kernel(x, y):
    # var = 1.0
    gamma = 1.0
    return math.exp(-gamma * (x - y)**2)


def rollout(node, goalPos, actions, rewards, config):
    '''
    Performs the sequence of actions in 'node.action_chain' using PINN models.\
    Chooses uniformly random values within the bounds for actions whose values are not supplied,\
    i.e., when the depth of the 'node' < maximum depth of the MCTS tree.
    
    # Parameters
    node    = a state in the MCTS exploration tree, \
              each node has a sequence of actions connecting it to the root node \
              stored as node.action_chain, \\
    goalPos = (x, y) coordinates of the goal, \\
    actions = list of actions taken so far, \\
    rewards = rewards corresponding to all previous actions, \\
    config  = configuration of the environment

    # Returns
    reward (adapted using GP-UCB if args.adaptive = True)
    '''
    action_chain = node.action_chain.copy()
    for i in range(node.depth, ACTION_PARAMETERS):
        action_chain.append(np.random.uniform(bnds[i][0], bnds[i][1]))
    reward = evalGP(goalPos, action_chain, actions, rewards, config)
    return reward


def kickstart(node, goalPos, actions, rewards, config):
    '''
    Build initial MCTS tree below 'node' with uniformly chosen actions from within the bounds. \
    Discretizing the continuous action space.
    
    # Parameters
    node    = a state in the MCTS exploration tree, \
              each node has a sequence of actions connecting it to the root node \
              stored as node.action_chain, \\
    goalPos = (x, y) coordinates of the goal, \\
    actions = list of actions taken so far, \\
    rewards = rewards corresponding to all previous actions, \\
    config  = configuration of the environment

    # Returns
    None (Just updates the tree below 'node')
    '''
    node.actions = [
        bnds[node.depth][0] + i *
        (bnds[node.depth][1] - bnds[node.depth][0]) / NUM_SPLITS
        for i in range(NUM_SPLITS + 1)
    ]
    for action in node.actions:
        node.num_actions.append(1)
        child_node = Node(node.depth + 1, node, node.action_chain.copy() + [action])
        node.children.append(child_node)
        reward = rollout(child_node, goalPos, actions, rewards, config)
        node.val_children.append(reward)


def MCTS(node, goalPos, actions, rewards, config):
    '''
    Performs discretized MCTS algorithm on the tree rooted in 'node'
    
    # Parameters
    node    = a state in the MCTS exploration tree, \
              each node has a sequence of actions connecting it to the root node \
              stored as node.action_chain, \\
    goalPos = (x, y) coordinates of the goal, \\
    actions = list of actions taken so far, \\
    rewards = rewards corresponding to all previous actions, \\
    config  = configuration of the environment

    # Returns
    reward
    '''
    
    # Simulation ----------
    if node.depth == ACTION_PARAMETERS:
        return rollout(node, goalPos, actions, rewards, config)
    max_val = -1e9
    max_idx = None
    A = len(node.children)
    if A == 0:

        # Expansion ----------
        kickstart(node, goalPos, actions, rewards, config)
        reward = rollout(node, goalPos, actions, rewards, config)

    # Progressive Widening ----------
    elif math.sqrt(sum(node.num_actions)) > A:
        new_action = np.random.uniform(bnds[node.depth][0],
                                       bnds[node.depth][1])
        node.actions.append(new_action)
        node.num_actions.append(1)
        new_node = Node(node.depth + 1, node,
                        node.action_chain.copy() + [new_action])
        node.children.append(new_node)
        reward = rollout(new_node, goalPos, actions, rewards, config)
    else:
        # Kernel Regression ----------
        W = [0.0 for _ in range(A)]
        V = [0.0 for _ in range(A)]
        for i in range(A):
            W[i] = sum([
                kernel(node.actions[i], node.actions[j]) * node.num_actions[j]
                for j in range(A)
            ])
            V[i] = sum([
                kernel(node.actions[i], node.actions[j]) *
                node.val_children[j] * node.num_actions[j] for j in range(A)
            ]) / sum([
                kernel(node.actions[i], node.actions[j]) * node.num_actions[j]
                for j in range(A)
            ])
        for i in range(A):
            
            # Selection using the UCB criterion ----------
            curr_val = V[i] + C * math.sqrt(math.log(sum(W)) / W[i])
            if curr_val > max_val:
                max_val = curr_val
                max_idx = i

        child_node = node.children[max_idx]
        reward = MCTS(child_node, goalPos, actions, rewards, config)

        # Backpropogation ----------
        node.val_children[max_idx] = (node.val_children[max_idx] * node.num_actions[max_idx] + reward) / (node.num_actions[max_idx] + 1)
        node.num_actions[max_idx] += 1
        node.val_children.append(reward)
    return reward


def get_best_action(node):
    '''
    Chooses the best sequence of action from the root node to 'node'
    
    # Parameters
    node    = a state in the MCTS exploration tree, \
              each node has a sequence of actions connecting it to the root node \
              stored as node.action_chain

    # Returns
    List of actions
    '''
    if node.depth == ACTION_PARAMETERS or len(node.children) == 0:
        return node.action_chain.copy()
    A = len(node.children)
    max_vis = -1e9
    max_val = -1e9
    max_idx = None
    for i in range(A):
        if node.num_actions[i] > max_vis or (node.num_actions[i] == max_vis and node.val_children[i] > max_val):
            max_vis = node.num_actions[i]
            max_val = node.val_children[i]
            max_idx = i
    return get_best_action(node.children[max_idx])


# *********************************************************** #
# ******************** RUN THE ALGORITHM ******************** #


# ---------- Prepare files to record results ----------
f = open('experiments/regret_result_' + args.env + '.txt', 'a')
if args.perception:
    f.write('Kernel_Widening_With_Perception\n')
else:
    f.write("Kernel_Widening_Without_Perception\n")
f.write("Action Parameters: %f, Contexts: %f, Actions: %f, D: %f\n" %
        (ACTION_PARAMETERS, NUM_CONTEXTS, NUM_ACTIONS, NUM_SPLITS))


img_dir = config['img_dir']
regrets = []
for iter in range(NUM_CONTEXTS):
    print(f"Iteration {iter+1}")

    # Generate and record a new context
    args.img_file = "%s/rgb_%d.png" % (img_dir, iter)
    generate_random_state(gym, sim, env, viewer, args, config)

    if args.perception:
        goalPos = get_goal_position(gym, sim, env, viewer, args, config)
    else:
        goalPos = get_goal_position_from_simulator(gym, sim, env, viewer, args, config)

    opt_reward = 1.0
    actions = []
    rewards = []
    best_reward = -math.inf
    for attempt in range(NUM_ACTIONS):
        if attempt != 0 and args.adaptive:
            gaussian_process.fit(actions, rewards)
        max_reward = -math.inf
        best_action = None

        root_node = Node()
        for _ in range(PINN_ATTEMPTS):
            MCTS(root_node, goalPos, actions, rewards, config)
            action = get_best_action(root_node)
            curr_size = len(action)
            for i in range(curr_size, ACTION_PARAMETERS):
                action.append(np.random.uniform(bnds[i][0], bnds[i][1]))
            reward = evalGP(goalPos, action, actions, rewards, config)
            if reward > max_reward:
                max_reward = reward
                best_action = action

        reward_ideal = execute_action(gym, sim, env, viewer, args, config, best_action)
        reward_curr = execute_action_pinn(config, goalPos, best_action)
        if reward_ideal > best_reward:
            best_reward = reward_ideal
        actions.append(best_action)
        rewards.append(reward_ideal - reward_curr)
    regret = (opt_reward - best_reward) / abs(opt_reward)
    if regret < 0.0:
        regret = 0.0

    print('Best Reward:', best_reward)
    print('Optimal Reward:', opt_reward)
    print('Regret:', regret)
    regrets.append(regret)

print('Final:', np.mean(regrets), np.std(regrets))
f.write("Final %f %f\n\n" % (np.mean(regrets), np.std(regrets)))
f.close()