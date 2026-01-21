from isaacgym import gymapi, gymutil
import ast
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor


np.random.seed(0)
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(custom_parameters=[
    {
        "name": "--contexts",
        "type": int,
        "default": 10,
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
        "name": "--adaptive",
        "type": ast.literal_eval,
        "default": True,
        "help": "Use GP-UCB"
    },
])

args.fast = False

if args.env == 'pendulum':
    from environments.pendulum_class import Env
elif args.env == 'sliding':
    from environments.sliding_class import Env
elif args.env == 'wedge':
    from environments.wedge_class import Env
elif args.env == 'sliding_bridge':
    from environments.sliding_bridge_class import Env
elif args.env == 'pendulum_wedge':
    from environments.pendulum_wedge_class import Env
else:
    raise Exception("Unknown Environment!")

ACTION_PARAMETERS = len(Env.bnds)
NUM_CONTEXTS = args.contexts
NUM_ACTIONS = args.actions
NUM_SPLITS = args.D

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

f = open('experiments/regret_result_' + args.env + '.txt', 'a')
f.write('MBPO_ORIG\n')
# if args.perception:
    # f.write('MBPO_With_Perception\n')
# else:
#     f.write("MBPO_Without_Perception\n")

f.write("Action Parameters: %f, Contexts: %f, Actions: %f\n" % (ACTION_PARAMETERS, NUM_CONTEXTS, NUM_ACTIONS))

# with open('experiments/position_' + args.env + '.txt', 'a') as pos_f:
#     if args.adaptive:
#         pos_f.write(f'PhyPlan_With_GP-UCB\n')
#     else:
#         pos_f.write(f'PhyPlan_Without_GP-UCB\n')

sim_params = gymapi.SimParams()
sim_params.dt = 1 / 200
sim_params.enable_actor_creation_warning = False
# if args.env == 'sliding_bridge':
#     sim_params.dt = 1 / 200

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
    else:
        main_cam_pos = gymapi.Vec3(1.5, -1.5, 2)
        main_cam_target = gymapi.Vec3(-0.5, -0.5, 0.0)

    gym.viewer_camera_look_at(viewer, None, main_cam_pos, main_cam_target)

# camera_properties = gymapi.CameraProperties()
# camera_properties.width = 500
# camera_properties.height = 500
# camera_handle = gym.create_camera_sensor(env, camera_properties)
# camera_position = gymapi.Vec3(0.0, 0.001, 2.1)
# if args.env == 'sliding_bridge':
#     camera_position = gymapi.Vec3(0.0, 0.001, 8.0)

# camera_position = gymapi.Vec3(0.0, 0.001, 2)
# camera_target = gymapi.Vec3(0, 0, 0)
# if args.env == 'sliding':
#     camera_position = gymapi.Vec3(-0.75, -0.649, 1.75)
#     camera_target = gymapi.Vec3(-0.75, -0.65, 0)
# elif args.env == 'pendulum':
#     camera_position = gymapi.Vec3(0, 0.001, 1.75)
# elif args.env == 'sliding_bridge':
#     camera_position = gymapi.Vec3(0.0, 0.001, 2.0)

# gym.set_camera_location(camera_handle, env, camera_position, camera_target)


# Define the Radial Basis Function (RBF)
def rbf(x, mu, Lambda_inv):
    # Compute the RBF value for input x, mean mu, and inverse covariance Lambda_inv
    diff = x - mu
    exponent = -0.5 * torch.matmul(torch.matmul(diff.T, Lambda_inv), diff)
    return torch.exp(exponent)


# Policy Network Class
class RBFPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_basis_functions, bounds):
        super(RBFPolicyNetwork, self).__init__()
        
        # Number of RBFs (n in the equation)
        self.num_basis_functions = num_basis_functions
        
        # Parameters for the RBFs: weights, centers (mu), and Lambda (inverse of covariance)
        self.weights = nn.Parameter(torch.randn(output_size, num_basis_functions, device=device))  # w_i
        self.centers = nn.Parameter(torch.randn(num_basis_functions, output_size, input_size, device=device))  # mu_i
        self.Lambda_inv = nn.Parameter(torch.eye(input_size, device=device).repeat(output_size, 1, 1).unsqueeze(0).repeat(num_basis_functions, 1, 1, 1))  # Lambda^-1 (diagonal covariance)

        self.bnds = torch.tensor(bounds, device=device).T

        self.dynamics_GP = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), n_restarts_optimizer=9)
        self.adapter_GP = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), n_restarts_optimizer=9)
                                #  GaussianProcessRegressor(kernel=RBF(length_scale=1.0), n_restarts_optimizer=9)]

    
    def forward(self, x):
        # Compute the policy π(x, θ) as a sum of weighted RBFs
        rbf_values = []
        for i in range(self.num_basis_functions):
            # Extract the center and Lambda_inv for the i-th RBF
            mu_i = self.centers[i]
            Lambda_inv_i = self.Lambda_inv[i]
            
            outputs = []
            for j in range(self.weights.shape[0]):
                # Compute RBF value for the i-th basis function
                rbf_val = rbf(x, mu_i[j], Lambda_inv_i[j])
                outputs.append(rbf_val)
            rbf_values.append(torch.stack(outputs).to(device))
        
        # Convert to tensor and compute the weighted sum
        rbf_values = torch.stack(rbf_values)
        policy_output = torch.matmul(self.weights, rbf_values).diag()

        policy_output = torch.tanh(policy_output)  # [-1, 1] range
        policy_output = (policy_output + 1) * 0.5 * (self.bnds[1] - self.bnds[0]) + self.bnds[0]  # Scale to [a, b]

        return policy_output
    
    def update_GP(self, actions, difference):
        self.adapter_GP.fit(actions, difference)
    
    def predict_GP(self, action):
        mean_pred, std_pred = self.adapter_GP.predict([action], return_std=True)
        return mean_pred, std_pred


def CustomLoss(state, goal, sigma=1):
    return 1 - torch.exp(-(torch.norm(state - goal)/sigma)**2)


# Hyperparameters
N = NUM_ACTIONS  # Number of epochs
E = 10  # Environment steps
if args.env == 'pendulum_wedge':
    E = 50
G = 100 # Gradient updates
D = 0  # Num rollouts for GP adaptation
T = NUM_CONTEXTS  # Number of Test Contexts
lr = 0.002  # Learning rate for policy updates

D_env = []
task_env = Env()
state_dim = 6
config = task_env.setup(gym, sim, env, viewer, args)
policy = RBFPolicyNetwork(state_dim, len(task_env.action_name), 50, task_env.bnds)
opt_reward = 1.0

# Optimizer
optimizer = optim.Adam(policy.parameters(), lr=lr)

actions = []
discrepancy = []

action_taken = []
fin_state = []

start = time.time()

# Main optimization loop
for epoch in range(N):
    # actions = []
    # discrepancy = []
    for step in range(E):
        # Sample state from environment (this should be your specific environment's state sampling)
        task_env.generate_random_state(gym, sim, env, viewer, args, config)

        ball_pos = task_env.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
        ball_height = task_env.get_piece_height(gym, sim, env, viewer, args, config)
        goal_pos = task_env.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
        goal_height = task_env.get_goal_height(gym, sim, env, viewer, args, config)

        current_state = [ball_pos[0], ball_pos[1], ball_height, goal_pos[0], goal_pos[1], goal_height]
        current_state_tensor = torch.tensor(current_state, device=device, dtype=torch.float32)

        # Take action and store in D_env (you'd interact with your environment here)
        D_env.append(current_state_tensor.unsqueeze(0))

        # Apply random controls and record data
        for iters in range(E):
            rand_action = task_env.generate_random_action()
            _, actual_state = task_env.execute_action(gym, sim, env, viewer, args, config, rand_action, return_ball_pos=True)
            ball_height = task_env.get_piece_height(gym, sim, env, viewer, args, config)
            actual_state = torch.tensor([actual_state[0], actual_state[1], ball_height])

            action_taken.append(torch.tensor(rand_action))
            fin_state.append(actual_state)

        # policy_output = policy(current_state_tensor)
        # pinn_state = task_env.get_final_state_via_PINN(config, policy_output)
        # _, actual_state = task_env.execute_action(gym, sim, env, viewer, args, config, policy_output, return_ball_pos=True)
        # ball_height = task_env.get_piece_height(gym, sim, env, viewer, args, config)
        # actual_state = torch.tensor([actual_state[0], actual_state[1], ball_height], device=device)
        # actions.append(policy_output.unsqueeze(0))
        # discrepancy.append((actual_state - pinn_state).unsqueeze(0))

    policy.dynamics_GP.fit(torch.stack(action_taken), torch.stack(fin_state))

    # actions = torch.cat(actions, dim=0)
    # discrepancy = torch.cat(discrepancy, dim=0)
    # policy.update_GP(actions.detach().cpu().numpy(), discrepancy.detach().cpu().numpy())

    # # Step 3: Perform rollouts in the learned model
    # for _ in range(M):
    #     # Sample state from D_env
    #     state, action = D_env[np.random.randint(0, len(D_env))]
    #     state_tensor = torch.tensor(state, device=device, dtype=torch.float32)
    #     ball_state, goal_state = state_tensor.chunk(2)
    #     state_final = task_env.get_final_state_via_PINN(config, policy_output[i])
    #     D_model.append((state, state_final))

    # Step 4: Update policy parameters using gradients
    for back in range(G):
        if len(D_env) > 0:
            states = torch.cat(D_env, dim=0)
            ball_state, goal_state = states.chunk(2, dim=1)

            # Compute policy loss
            loss = 0
            for i, state in enumerate(states):
                policy_output = policy(state)
                # state_final = task_env.get_final_state_via_PINN(config, policy_output)
                state_final = torch.tensor(policy.dynamics_GP.predict([policy_output.cpu().detach().numpy()]), device=device, requires_grad=True)
                if (args.adaptive and len(actions) > 0):
                    mean_correction, std_correction = policy.predict_GP(policy_output.detach().cpu().numpy())
                    state_final += torch.tensor(mean_correction + 0.5 * std_correction, device=device).squeeze(0)
                loss += CustomLoss(state_final, goal_state[i], sigma=0.8)
            loss /= len(states)

            # Update policy parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss {back}:", loss.cpu().detach().numpy())

    if (args.adaptive):
        # id = np.random.randint(0, len(D_env))
        ids = np.random.choice(np.arange(0, len(D_env)), size=D, replace=False)
        for id in ids:
            policy_output = policy(D_env[id].squeeze(0))
            pinn_state = task_env.get_final_state_via_PINN(config, policy_output)
            _, actual_state = task_env.execute_action(gym, sim, env, viewer, args, config, policy_output, return_ball_pos=True)
            ball_height = task_env.get_piece_height(gym, sim, env, viewer, args, config)
            actual_state = torch.tensor([actual_state[0], actual_state[1], ball_height], device=device)
            actions.append(policy_output.unsqueeze(0))
            discrepancy.append((actual_state - pinn_state).unsqueeze(0))

        if len(ids) > 0:
            policy.update_GP(torch.cat(actions, dim=0).detach().cpu().numpy(), torch.cat(discrepancy, dim=0).detach().cpu().numpy())

    print(f"Epoch {epoch + 1}/{N} completed.")
    
    # rewards = []
    # regrets = []
    # for i in range(T):
    #     task_env.generate_random_state(gym, sim, env, viewer, args, config)
    #     ball_pos = task_env.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
    #     ball_height = task_env.get_piece_height(gym, sim, env, viewer, args, config)
    #     goal_pos = task_env.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
    #     goal_height = task_env.get_goal_height(gym, sim, env, viewer, args, config)
    #     state = [ball_pos[0], ball_pos[1], ball_height, goal_pos[0], goal_pos[1], goal_height]
    #     state = torch.tensor(state, device=device, dtype=torch.float32)
        
    #     action = policy(state)
    #     reward = task_env.execute_action(gym, sim, env, viewer, args, config, action)
    #     regret = (opt_reward - reward) / abs(opt_reward)
    #     rewards.append(reward)
    #     regrets.append(regret)

    # print('Final:', np.mean(regrets), np.std(regrets))
    # f.write("Final %f %f\n\n" % (np.mean(regrets), np.std(regrets)))
    # f.close()

end = time.time()

rewards = []
regrets = []
for i in range(T):
    task_env.generate_random_state(gym, sim, env, viewer, args, config)
    ball_pos = task_env.get_piece_position_from_simulator(gym, sim, env, viewer, args, config)
    ball_height = task_env.get_piece_height(gym, sim, env, viewer, args, config)
    goal_pos = task_env.get_goal_position_from_simulator(gym, sim, env, viewer, args, config)
    goal_height = task_env.get_goal_height(gym, sim, env, viewer, args, config)
    state = [ball_pos[0], ball_pos[1], ball_height, goal_pos[0], goal_pos[1], goal_height]
    state = torch.tensor(state, device=device, dtype=torch.float32)
    
    action = policy(state)
    reward = task_env.execute_action(gym, sim, env, viewer, args, config, action)
    regret = (opt_reward - reward) / abs(opt_reward)
    rewards.append(reward)
    regrets.append(regret)

print('Final:', np.mean(regrets), np.std(regrets))
f.write("Final %f %f\n" % (np.mean(regrets), np.std(regrets)))
f.write("Time %f\n\n" % (end - start))
f.close()
