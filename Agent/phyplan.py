import os
from isaacgym import gymapi, gymutil
import ast
import copy
import math
import torch
import torch.nn as nn
import multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mcts_agent_softgreedy import MCTS_Agent
import itertools
import time
from scipy.stats import entropy
# from plot_vis_i_handler import write_plot_vis_i, read_plot_vis_i
from io import BytesIO
import threading
import imageio
from ui_bridge import update_label

# global plot_vis_i
# plot_vis_i = 0
# write_plot_vis_i(plot_vis_i)

# mp.set_start_method('spawn')

args = gymutil.parse_arguments(custom_parameters=[
    {
        "name": "--vis",
        "type": ast.literal_eval,
        "default": False,
        "help": "Render a Visualization of Tool Selection"
    },
    {
        "name": "--hist",
        "type": ast.literal_eval,
        "default": False,
        "help": "Render histograms of state probabilities as predicted by GFN"
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
        "default": 25,
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
    {
        "name": "--epochs",
        "type": int,
        "default": 10,
        "help": "epochs for GFlowNet"
    },
    {
        "name": "--lr",
        "type": float,
        "default": 1e-1,
        "help": "learning rate for GFlowNet"
    },
    {
        "name": "--gamma",
        "type": float,
        "default": 0.4,
        "help": "weight for Uniform distribution"
    },
    {
        "name": "--topk",
        "type": int,
        "default": 3,
        "help": "number of tools to evaluate after training GFlowNet"
    },
    {
        "name": "--tol",
        "type": float,
        "default": 0,
        "help": "Stopping criterion for regret by topk tools"
    },
    {
        "name": "--energy",
        "type": int,
        "default": 4,
        "help": "sensitivity to reward function"
    }
])

args.fast = False

np.random.seed(0)

tool_num_dict = {'pendulum': [4, 6], 'wedge': [6], 'bridge': [4, 6]}

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
    if args.env == 'sliding':
        main_cam_pos = gymapi.Vec3(1.5, -1.5, 2)
        main_cam_target = gymapi.Vec3(-0.5, -0.5, 0.0)
    if args.env == 'pendulum_wedge':
        main_cam_pos = gymapi.Vec3(-1, 1, 2)
        main_cam_target = gymapi.Vec3(0.2, -0.2, 0.0)

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

if args.vis:
    img_buf = BytesIO()
    imageio.imwrite(img_buf, np.reshape(gym.get_camera_image(sim, env, config["cam_handle"], gymapi.IMAGE_COLOR), (config["img_height"], config["img_width"], 4))[:, :, :3], format='png')
    update_label("IsaacGym", img_buf)
    update_label("Context", "Contexts: __/__")
    update_label("PredRew", "Pred. Reward: ____")
    update_label("Trial", "Trial: __/__")
    update_label("IdealRew", "Ideal Reward: ____")
    update_label("Tool", "Tool Combination: ____")
    update_label("ChosAct", "Chosen Action: ____")


agent = MCTS_Agent(gym, sim, env, viewer, args, task_env, config, task_env.bnds)

def eval_reward(goal_pos, tools, args, execute_action=False):
    print("Tools:", tools)
    update_label("Tool", "Tool Combination: " + str(tools))
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

class Forward_policy(nn.Module): 
    def __init__(self,in_size,out_size):
        '''
        FNN to learn forward flow F_theta
        
        # Parameters
        in_size     = Input size of FNN, \\
        out_size    = Output size of FNN 
        
        '''
        super(Forward_policy, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)  

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x
    
class z0(nn.Module):
    def __init__(self,in_size):
        '''
        FNN to learn scalar inflow Z_theta to Flow Network
        
        # Parameters
        in_size     = Input size of FNN, \\
        
        '''
        super(z0, self).__init__()
        self.fc1 = nn.Linear(in_size, 1) 
        # self.fc2 = nn.Linear(4, 1) 
        # nn.init.uniform_(self.fc1.weight, a=-100, b=-98)
        # nn.init.uniform_(self.fc1.bias, a=-100, b=-98)  

    def forward(self, x):
        x = 1000*F.sigmoid(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x

states=[]
probs=[]   
class Tool_Selector_GFN():
    def __init__(self, options, args, execute_action):
        self.ACTION_PARAMETERS = len(options)
        self.max_actions = max(len(option) for option in options)
        self.tools = options
        self.args = args
        self.execute_action = execute_action
        self.epochs = args.epochs
        self.lr = args.lr
        self.gamma = args.gamma
        self.energy = args.energy
        
        self.variable_list = []
        self.best_regrets = []
        self.plot_vis_i = 0

    def plot_histogram(self, states, probs, epoch):
        plt.figure(figsize=(8, 8), dpi=200)
        probs_values = [p.detach().cpu().item() for p in probs]
        sorted_indices = np.argsort(probs_values)[::-1]  # Sort indices in descending order
        topk_indices = sorted_indices[:self.args.topk]  # Get the top-k indices
        if self.args.vis==True:
            colors = ['red' if i in topk_indices else 'blue' for i in range(len(probs_values))]
        elif self.args.hist==True:
            colors = ['blue' if i in topk_indices else 'blue' for i in range(len(probs_values))]
        # print(states)
        # print(probs_values)
        # print(colors)
        sorted_pairs = sorted(zip(states, probs, colors))  
        sorted_state, sorted_prob, sorted_colors = zip(*sorted_pairs)
        plt.barh(range(len(sorted_prob)), [p.detach().cpu().item() for p in sorted_prob], tick_label=[str(a) for a in sorted_state], color=sorted_colors)
        if epoch > 0:
            plt.title(f"GFN Tool Priorities - Epoch {epoch}", fontsize='30')
        else:
            plt.title(f"GFN Tool Priorities - Untrained", fontsize='30')
        plt.ylabel("Tool Constructions", fontsize='20')
        plt.xlabel("Probability", fontsize='20')
        plt.margins(y=0)
        # plt.xticks(rotation=90, ha="right")
        # plt.xticks(rotation=90, ha="right", fontsize='20')
        # plt.xticks(rotation=90, ha="right")
        plt.yticks(fontsize='0')
        plt.tight_layout()
        # plt.savefig(f"vis_images/{plot_vis_i}_GFN.png", dpi=100)  # Save histogram with sequence number
        if self.args.vis:
            img_buf = BytesIO()
            plt.savefig(img_buf, dpi=200)
            update_label("GFN", img_buf)

        if self.args.hist:
            os.makedirs("vis_images", exist_ok=True)
            plt.savefig(f"vis_images/{self.plot_vis_i}_GFN.png", dpi=200)
            self.plot_vis_i+=1
        plt.close()

    def plot_histogram_with_highlight(self, states, probs, selected_tool):
        """
        Creates a histogram highlighting a specific selected tool.

        :param states: List of tool combinations (states).
        :param probs: List of probabilities corresponding to the states.
        :param selected_tool: The specific tool combination to highlight.
        :param epoch: The current epoch for labeling the plot.
        """
        probs_values = [p.detach().cpu().item() for p in probs]
        colors = ['red' if state == selected_tool else 'blue' for state in states]

        sorted_pairs = sorted(zip(states, probs, colors))  
        sorted_state, sorted_prob, sorted_colors = zip(*sorted_pairs)
        plt.figure(figsize=(8, 8), dpi=200)
        plt.barh(range(len(probs_values)), [p.detach().cpu().item() for p in sorted_prob], tick_label=[str(a) for a in sorted_state], color=sorted_colors)
        plt.title(f"Trained GFN Tool Priorities \n Currently Selected Tool", fontsize='30')
        plt.ylabel("Tool Constructions", fontsize='20')
        plt.xlabel("Probability", fontsize='20')
        plt.margins(y=0)
        # plt.xticks(rotation=90, ha="right")
        # plt.xticks(rotation=90, ha="right", fontsize='20')
        # plt.xticks(rotation=90, ha="right")
        plt.yticks(fontsize='0')
        plt.tight_layout()
        # plt.savefig(f"vis_images/{plot_vis_i}_GFN.png", dpi=100)  # Save histogram with sequence number
        img_buf = BytesIO()
        plt.savefig(img_buf, dpi=200)
        update_label("GFN", img_buf)
        plt.close()

    def priority(self, model):
        priority_states = []
        ind = [range(len(lst)) for lst in self.tools]
        poss_ind = list(itertools.product(*ind))
        poss_tools = list(itertools.product(*self.tools))
        for i in range(len(poss_ind)):
            score = 1
            s_t = torch.full((self.ACTION_PARAMETERS,), -1).to(dtype=torch.float32)
            for j in range(self.ACTION_PARAMETERS):
                score *= model(s_t)[poss_ind[i][j]]/torch.sum(model(s_t)[:len(self.tools[j])])
                s_t[j] = poss_ind[i][j]
            priority_states.append((poss_tools[i],score))
        priority_states.sort(key=lambda x: x[1], reverse=True)
        sorted_states = [state[0] for state in priority_states]
        rewards = [state[1] for state in priority_states]
        return sorted_states, rewards

    def GFN(self, goal_pos):
        pol_f = Forward_policy(self.ACTION_PARAMETERS, self.max_actions)
        z = z0(self.ACTION_PARAMETERS)

        optimizer = torch.optim.Adam([
            {'params': pol_f.parameters(), 'lr': self.lr},
            {'params': z.parameters(), 'lr': self.lr}
        ])
        epsilon = 1e-16
        tot_loss = 0
        start_time = time.time()
        state, prob =self.priority(pol_f)
        states.append(state)
        probs.append(prob)
        if args.vis or args.hist:
            self.plot_histogram(states[-1], probs[-1], 0)
        for epoch in range(self.epochs):
            loss = 0
            s_t = torch.full((self.ACTION_PARAMETERS,), -1, dtype=torch.float32).requires_grad_()
            loss += torch.log(z(s_t)+epsilon)
            for step in range(self.ACTION_PARAMETERS):
                flow_f = (1 - self.gamma) * (pol_f(s_t))
                uniform_dist = torch.rand(len(self.tools[step]))
                flow_f[:len(self.tools[step])] += self.gamma * uniform_dist
                flow_f[len(self.tools[step]):] = torch.zeros(self.max_actions - len(self.tools[step]))

                af_flow = (flow_f) / (flow_f).sum()
                af = torch.distributions.Categorical(probs=af_flow, validate_args=False).sample()

                s_new = []
                for cell in range(self.ACTION_PARAMETERS):
                    if cell == step:
                        s_new.append(af)
                    else:
                        s_new.append(s_t[cell].item())
                s_t = torch.tensor(s_new, dtype=torch.float32, requires_grad=True)

                p_f = af_flow[af]
                loss += torch.log(p_f + epsilon)

                if step == self.ACTION_PARAMETERS - 1:
                    tool_list = [self.tools[t][int(s_t[t])] for t in range(self.ACTION_PARAMETERS)]

                    update_label("ChosAct", "Chosen Action: ____")
                    update_label("PredRew", "Pred. Reward: ____")
                    rew, _ = eval_reward(goal_pos, tool_list, self.args, execute_action=False)
                    loss -= torch.log(torch.exp(torch.tensor(self.energy*rew, dtype=torch.float32, requires_grad=True)) + epsilon)

            tot_loss += loss ** 2
            if epoch % 1 == 0:
                print(f'Epoch {epoch + 1}, Loss: {tot_loss.item():.5f}')
                optimizer.zero_grad()
                tot_loss.backward()
                optimizer.step()
                tot_loss = 0
            
            state, prob =self.priority(pol_f)
            states.append(state)
            probs.append(prob)
            if args.vis or args.hist:
                self.plot_histogram(states[-1], probs[-1], epoch + 1)
        end_time = time.time()
        print(f"Total time taken for {self.epochs} epochs: {end_time - start_time:.2f} seconds")
        return pol_f, z 

    def search(self, goal_pos):
        pol_f, _ = self.GFN(goal_pos)   
        return self.priority(pol_f) 
    

args_copy = copy.deepcopy(args)
args_copy.actions = 1
args_copy.adaptive = False
args_copy.simulate = False

selector = Tool_Selector_GFN(options=task_env.available_tools, args=args_copy, execute_action=False)

img_dir = config['img_dir']
regrets = []
action_wise_regrets = [[] for _ in range(NUM_ACTIONS)]
entropies = []
best_inds = [-1] * NUM_CONTEXTS
# pinn_rewards = []
for iter in range(NUM_CONTEXTS):
    print(f"Context {iter+1}")
    update_label("Context", "Context: " + str(iter + 1) + "/" + str(NUM_CONTEXTS))
    update_label("PredRew", "Pred. Reward: ____")
    update_label("Trial", "Trial: __/__")
    update_label("IdealRew", "Ideal Reward: ____")
    update_label("Tool", "Tool: ____")
    update_label("ChosAct", "Chosen Action: ____")
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

    priority_list, prob = selector.search(goal_pos)
    entropies.append(entropy(np.array([p.detach().cpu().item() for p in prob]),base=2))
    agent.best_regrets = []
    agent.args.img_file = "%s/rgb_%d.png" % (img_dir, iter)
    min_regret = 1e9
    best_tool = None
    start_time = time.time()
    for i in range(args.topk):
        tools=list(priority_list[i])
        if args.vis:
            selector.plot_histogram_with_highlight(states[-1], probs[-1], priority_list[i])
        reward_t, regret_t = eval_reward(goal_pos, tools, args, execute_action=True)
        if regret_t<=min_regret:
            best_inds[iter] = i
            min_regret = regret_t
            reward = reward_t
            best_tool = priority_list[i]
            # print(tools)
            best_regrets = copy.deepcopy(agent.best_regrets)[-NUM_ACTIONS:]
            # pinn_rews = [agent.variable_list[-2][-1-n][-1] for n in range(NUM_ACTIONS)]
            # if i == 0:
            #     pinn_rewards.append(max(pinn_rews))
            # else:
            #     pinn_rewards[iter] = max(pinn_rews)
        # print(pinn_rewards)
        if min_regret<=args.tol:
            break
    end_time = time.time()
    # pinn_regrets = [(1 - reward) for reward in pinn_rewards]
    print(f"Total time taken for {args.topk} executions of 5 trials each : {end_time - start_time:.2f} seconds")
    regret = min_regret
    # reward, regret = call_eval_proc(goal_pos, tools, args)
    # result = call_eval_proc(goal_pos, tools, args, get_intermediate_regret=True, execute_action=True)
    # reward, regret = result[0], result[1]
    print("**********************************************")
    print("Tools:", best_tool)
    print("Tool Reward:", reward)
    print("Final Regret:", regret)
    print("**********************************************")
    regrets.append(regret)
    for act in range(NUM_ACTIONS):
        action_wise_regrets[act].append(best_regrets[act])

# print(f'Final:', np.mean(regrets), np.std(regrets))

f = open('experiments/regret_result_gfn_rest_tool_select_' + args.env + '.txt', 'a')
for i in range(NUM_ACTIONS):
    f.write("\nGFN_Selected_Tool_(top-%d)\n" % (args.topk))
    f.write("Contexts: %f, Actions: %f, D: %f\n" % (NUM_CONTEXTS, i+1, NUM_SPLITS))

    print(f'Final {i}:', np.mean(action_wise_regrets[i]), np.std(action_wise_regrets[i]))
    f.write("Final %f %f \n\n" % (np.mean(action_wise_regrets[i]), np.std(action_wise_regrets[i])))
    # print(f'Final {i}:', np.mean([pinn_regrets][i]), np.std([pinn_regrets][i]))
    # f.write("Final %f %f \n\n" % (np.mean([pinn_regrets][i]), np.std([pinn_regrets][i])))
# f.write("Entropy %f %f" % (np.mean(entropies), np.std(entropies)))


# sorted_states = []
# sorted_probs = []

# for i in range(len(states)): 
#     sorted_pairs = sorted(zip(states[i], probs[i]))  
#     sorted_state, sorted_prob = zip(*sorted_pairs) 
#     sorted_states.append(list(sorted_state))
#     sorted_probs.append(list(sorted_prob))

# states=sorted_states
# probs=sorted_probs

# for state in states:
#     f.write(",".join(map(str, state)) + "\n")
# probs_list = [[prob.item() for prob in probs[i]] for i in range(len(probs))]
# for prob in probs_list:
#     f.write(",".join(map(str, prob)) + "\n")
f.close()