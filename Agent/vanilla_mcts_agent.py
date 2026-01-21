import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

class Node():
    def __init__(self, depth=0, parent=None, action_chain=[]):
        self.parent = parent
        self.depth = depth
        self.children = []
        self.actions = []
        self.num_actions = []
        self.val_children = []
        self.action_chain = action_chain

    def copy(self):
        newNode = Node(depth=self.depth)
        if self.parent is not None:
            newNode.parent = self.parent.copy()
        newNode.children = self.children.copy()
        newNode.actions = self.actions.copy()
        newNode.num_actions = self.num_actions.copy()
        newNode.val_children = self.val_children.copy()
        newNode.action_chain = self.action_chain.copy()
        return newNode


class MCTS_Agent():
    def __init__(self, gym, sim, env, viewer, args, task_env, config, bounds):
        self.ACTION_PARAMETERS = len(bounds)
        self.PINN_ATTEMPTS = args.pinn_attempts
        self.NUM_CONTEXTS = args.contexts
        self.NUM_ACTIONS = args.actions
        self.NUM_SPLITS = args.D
        self.C = 1.0
        self.pinn_regrets = []
        self.pinn_counts = []
        self.variable_list = []
        self.best_regrets = []

        self.gym = gym
        self.sim = sim
        self.env = env
        self.args = args
        self.bnds = bounds
        self.viewer = viewer
        self.config = config
        self.task_env = task_env

        # scaler = StandardScaler()
        self.kernel = 1 * RBF(length_scale=1.0)
        # self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9, random_state=1234)
        self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9)
        # self.gaussian_process = make_pipeline(scaler, self.gaussian_process)

    def updateBounds(self, bounds):
        self.bnds = bounds

    def updateTask(self, task_env):
        self.task_env = task_env
    
    def updateConfig(self, config):
        self.config = config

    def updateArgs(self, args):
        self.args = args
        self.PINN_ATTEMPTS = args.pinn_attempts
        self.NUM_CONTEXTS = args.contexts
        self.NUM_ACTIONS = args.actions
        self.NUM_SPLITS = args.D

    def evalGP(self, goal_pos, action, actions, rewards, beta=0.5):
        if len(actions) == 0:
            return self.task_env.execute_action(self.gym, self.sim, self.env, self.viewer, self.args, self.config, action)
        if self.args.adaptive:
            # beta = 0
            mean_pred, std_pred = self.gaussian_process.predict([action], return_std=True)
            return self.task_env.execute_action(self.gym, self.sim, self.env, self.viewer, self.args, self.config, action) + mean_pred[0] + beta * std_pred[0]
        else:
            return self.task_env.execute_action(self.gym, self.sim, self.env, self.viewer, self.args, self.config, action)

    def rollout(self, node, goal_pos, actions, rewards):
        action_chain = node.action_chain.copy()
        for i in range(node.depth, self.ACTION_PARAMETERS):
            action_chain.append(np.random.uniform(self.bnds[i][0], self.bnds[i][1]))
        reward = self.evalGP(goal_pos, action_chain, actions, rewards)
        return reward

    def kickstart(self, node, goal_pos, actions, rewards):
        node.actions = [
            self.bnds[node.depth][0] + i *
            (self.bnds[node.depth][1] - self.bnds[node.depth][0]) / self.NUM_SPLITS
            for i in range(self.NUM_SPLITS + 1)
        ]
        for action in node.actions:
            node.num_actions.append(1)
            child_node = Node(node.depth + 1, node, node.action_chain.copy() + [action])
            node.children.append(child_node)
            reward = self.rollout(child_node, goal_pos, actions, rewards)
            node.val_children.append(reward)

    def MCTS(self, node, goal_pos, actions, rewards):
        if node.depth == self.ACTION_PARAMETERS:
            return self.rollout(node, goal_pos, actions, rewards)
        max_val = -1e9
        max_idx = None
        A = len(node.children)
        if A == 0:
            self.kickstart(node, goal_pos, actions, rewards)
            reward = self.rollout(node, goal_pos, actions, rewards)
        elif self.args.widening and math.sqrt(sum(node.num_actions)) > A:
            '''This does not look like widening!'''
            print("widening")
            new_action = np.random.uniform(self.bnds[node.depth][0], self.bnds[node.depth][1])
            node.actions.append(new_action)
            node.num_actions.append(1)
            new_node = Node(node.depth + 1, node, node.action_chain.copy() + [new_action])
            node.children.append(new_node)
            reward = self.rollout(new_node, goal_pos, actions, rewards)
            node.val_children.append(reward)
        else:
            for i in range(A):
                curr_val = node.val_children[i] + self.C * math.sqrt(math.log(sum(node.num_actions)) / node.num_actions[i])
                if curr_val > max_val:
                    max_val = curr_val
                    max_idx = i
            child_node = node.children[max_idx]
            reward = self.MCTS(child_node, goal_pos, actions, rewards)
            node.val_children[max_idx] = (
                node.val_children[max_idx] * node.num_actions[max_idx] +
                reward) / (node.num_actions[max_idx] + 1)
            node.num_actions[max_idx] += 1
        return reward

    def get_best_action(self, node):
        if node.depth == self.ACTION_PARAMETERS or len(node.children) == 0:
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
        return self.get_best_action(node.children[max_idx])

    def count(self, node):
        ans = 0
        for child in node.children:
            ans += self.count(child)
        return ans + len(node.children)

    def search(self, goal_pos, execute_action=False):
        opt_reward = 1.0
        actions = []
        rewards = []
        best_reward = -math.inf
        
        for attempt in range(self.NUM_ACTIONS):
            while len(self.pinn_regrets) <= attempt:
                self.pinn_regrets.append([])
                self.pinn_counts.append([])

            if attempt != 0 and self.args.adaptive:
                self.gaussian_process.fit(actions, rewards)
            max_reward = -math.inf
            best_action = None

            root_node = Node()
            # best_node_values = []
            # confidence_values = []
            # prev_reward = -2
            # second_prev_reward = -3
            prev_conf = -200
            # second_prev_conf = -300
            # prev_conf = [-2]*self.ACTION_PARAMETERS
            # second_prev_conf = [-3]*self.ACTION_PARAMETERS
            for pa in range(self.PINN_ATTEMPTS):
                self.MCTS(root_node, goal_pos, actions, rewards)
                action = self.get_best_action(root_node)
                curr_size = len(action)
                for i in range(curr_size, self.ACTION_PARAMETERS):
                    action.append(np.random.uniform(self.bnds[i][0], self.bnds[i][1]))
                reward = self.evalGP(goal_pos, action, actions, rewards)
                if reward > max_reward:
                    max_reward = reward
                    best_action = action

                pinn_regret = (opt_reward - max_reward) / abs(opt_reward)
                if (len(self.pinn_regrets[attempt]) <= pa):
                    self.pinn_regrets[attempt].append(pinn_regret)
                    self.pinn_counts[attempt].append(1)
                else:
                    self.pinn_regrets[attempt][pa] = (self.pinn_counts[attempt][pa] * self.pinn_regrets[attempt][pa] + pinn_regret) / (self.pinn_counts[attempt][pa] + 1)
                    self.pinn_counts[attempt][pa] += 1
                
                # avg_confidence = [0 for _ in range(self.ACTION_PARAMETERS)]
                # max_confidence = np.zeros(self.ACTION_PARAMETERS)
                # min_confidence = [np.Inf for _ in range(self.ACTION_PARAMETERS)]
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
                        # avg_confidence[i] += action_node.val_children[child]
                        # if action_node.val_children[child] > max_confidence[i]:
                        if action_node.val_children[child] > max_conf:
                            max_conf = action_node.val_children[child]
                            child_node = action_node.children[child]
                        # min_confidence[i] = min(min_confidence, action_node.val_children[child])
                    action_node = child_node.copy()
                    max_confidence += max_conf
                    # avg_confidence[i] /= num_children
                    # confidence_values.append(np.array([avg_confidence, max_confidence, min_confidence]))
                max_confidence /= self.ACTION_PARAMETERS

                # if max_reward < 1 and max_confidence < 1:
                if len(self.variable_list) == 0:
                    self.variable_list.append([])
                    self.variable_list.append([])
                    self.variable_list.append([])
                self.variable_list[0].append([max_confidence, max_reward])

                if not ready:
                    continue

                # if self.args.env != 'sliding_bridge':
                #     if abs(max_confidence - prev_conf) < 1.0e-4:
                #         break
                # else:
                #     if abs(max_confidence - prev_conf) < 1.0e-4 and abs(prev_conf - second_prev_conf) < 1.0e-4:
                #         break

                # if abs(max_confidence - prev_conf) < 1.0e-4:
                #     break

                # if (abs(max_confidence - prev_conf) < 1.0e-4).all():
                #     break
                # second_prev_conf = prev_conf
                prev_conf = max_confidence

                # best_node_values.append(reward)
                # if abs(reward - prev_reward) < 1.0e-4 and abs(prev_reward - second_prev_reward) < 1.0e-4:
                # if abs(reward - prev_reward) < 1.0e-4 and abs(max_confidence - prev_conf) < 1.0e-4:
                #     break
                # second_prev_reward = prev_reward
                # prev_reward = reward

            # self.variable_list.append(pa+1)

            # confidence_values = np.array(confidence_values)
            # plt.plot(confidence_values[:,0], label='avg_confidence', color='blue', marker='o')
            # plt.plot(confidence_values[:,1], label='max_confidence', color='red', marker='x')
            # plt.plot(confidence_values[:,2], label='min_confidence', color='green', marker='s')
            # # plt.plot(np.diff(confidence_values[:,1], 1), label='diff_confidence', color='black', marker='+')
            # plt.xlabel('Trials')
            # plt.ylabel('Confidence')
            # # plt.legend()
            # plt.title(f'MCTS confidence in {self.args.env}')
            # plt.savefig(f'plots/confidence_{self.args.env}_{attempt}.png', bbox_inches='tight')
            # # plt.show()
            # plt.clf()

            # plt.plot(best_node_values)
            # plt.xlabel('Iterations')
            # plt.ylabel('Best Node Value')
            # plt.title('Convergence of MCTS')
            # plt.show()

            # print('---- MCTS DATA ----')
            # print(root_node.actions)
            # print(root_node.num_actions)
            # print(root_node.val_children)
            # print('---- END ----')

            reward_curr = self.task_env.execute_action(self.gym, self.sim, self.env, self.viewer, self.args, self.config, best_action)
            if reward_curr > best_reward:
                best_reward = reward_curr

            self.best_regrets.append((opt_reward - best_reward) / abs(opt_reward))

        # plt.plot(variable_list)
        # plt.xlabel('Trials')
        # plt.ylabel('Iteration with PINN')
        # plt.title('PINN-MCTS Iterations to converge')
        # plt.savefig(f'plots/convergence_{self.args.env}_{goal_pos}.png')
        # # plt.show()

        regret = (opt_reward - best_reward) / abs(opt_reward)
        if regret < 0.0:
            regret = 0.0

        with open('experiments/position_' + self.args.env + '.txt', 'a') as pos_f:
            pos_f.write('\n')

        return best_reward, opt_reward, regret
