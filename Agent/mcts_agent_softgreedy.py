import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from graphviz import Digraph
from io import BytesIO
from ui_bridge import update_label
import time
import copy


class Node():
    def __init__(self, depth=0, parent=None, action_chain=[]):
        self.parent = parent
        self.depth = depth
        self.children = []
        self.actions = []
        self.num_actions = []
        self.val_children = []
        self.reward_history = []
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
        # newNode.reward_history = [r.copy() for r in self.reward_history]
        newNode.reward_history = copy.deepcopy(self.reward_history)
        return newNode


class MCTS_Agent():
    def __init__(self, gym, sim, env, viewer, args, task_env, config, bounds):
        self.ACTION_PARAMETERS = len(bounds)
        if len(bounds) >= 0: # Temporary, use 5 otherwise
            self.use_softmax = True
        else:
            self.use_softmax = False
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

    def visualize_tree(self, root):
        """
        Visualize the MCTS tree using graphviz.

        :param root: The root node of the tree.
        :param filename: The filename to save the visualization.
        """
        dot = Digraph(format='png')
        # dot.attr(size="1, 1", ratio="compress", dpi="1200")
        dot.attr(size="8, 1", ratio="compress", dpi="160")
        self._add_node(dot, root, 0, 0, 0, True)
        img_buf = BytesIO(dot.pipe(format='png'))
        update_label("MCTS", img_buf)

    def _add_node(self, dot, node, ucb, rew, exploration, condition):
        """
        Recursively add nodes and edges to the graphviz Digraph.

        :param dot: The graphviz Digraph object.
        :param node: The current node to add.
        """
        # print(node.depth)
        # print(node.val_children)
        # print(node.actions)
        # input()
        # if len(node.actions) > 0:
        #     node_label = f"Depth: {node.depth + 1}\nReward_mean: {str(round(np.mean(node.val_children).astype(float), 2)) if len(node.val_children) > 0 else str(round(rew, 2))}\nA_min: {str(round(node.actions[0], 2))}, A_max: {str(round(node.actions[0], 2))}"
        # else:
        #     node_label = f"Depth: {node.depth + 1}\nReward_mean: {str(round(np.mean(node.val_children).astype(float), 2)) if len(node.val_children) > 0 else str(round(rew, 2))}"
        node_label = f"{str(round(np.mean(node.val_children).astype(float), 2)) if len(node.val_children) > 0 else str(round(rew, 2))}"
        rew_ = f"{str(round(np.mean(node.val_children).astype(float), 2)) if len(node.val_children) > 0 else str(round(rew, 2))}"
        if node.depth == 0:
            node_label = "Root"
        else:
            node_label = str(round(ucb, 2)) + "\n(" + rew_ + ", " + str(round(exploration, 2)) + ")"
        if condition:
            dot.node(str(id(node)), label=node_label, color="blue", fontcolor="white", fillcolor="darkgreen", style="filled")
        else:
            dot.node(str(id(node)), label=node_label)
        if len(node.val_children) > 0:
            max_i = -1
            max_val = float('-inf')
            for i in range(len(node.children)):
                curr = node.val_children[i] + self.C * math.sqrt(math.log(sum(node.num_actions)) / node.num_actions[i])
                if curr > max_val:
                    max_val = curr
                    max_i = i
        for i, child in enumerate(node.children):
            ucb = node.val_children[i] + self.C * math.sqrt(math.log(sum(node.num_actions)) / node.num_actions[i])
            exploration = self.C * math.sqrt(math.log(sum(node.num_actions)) / node.num_actions[i])
            if condition and len(node.val_children) > 0 and i == max_i:
                dot.edge(str(id(node)), str(id(child)), color="blue", penwidth="2.0")
                self._add_node(dot, child, ucb, node.val_children[i], exploration, True)
            else:
                dot.edge(str(id(node)), str(id(child)))
                self._add_node(dot, child, ucb, 0 if len(node.val_children) == 0 else node.val_children[i], exploration, False)

    def updateBounds(self, bounds):
        self.ACTION_PARAMETERS = len(bounds)
        self.bnds = bounds
        if len(bounds) >= 0: # Temporary, use 5 otherwise
            self.use_softmax = True
        else:
            self.use_softmax = False

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

    def softmax_value(self, rewards, tau=1.0):
        rewards = np.array(rewards)
        exps = np.exp(tau * (rewards - np.max(rewards)))
        softmax_weights = exps / np.sum(exps)
        return np.sum(softmax_weights * rewards)

    def evalGP(self, goal_pos, action, actions, rewards, beta=0.5):
        ####################### Just to test execution, not needed otherwise so remove later #######################
        # fell_on_wedge = [False]
        # self.task_env.execute_action_pinn(self.config, goal_pos, action, fell_on_wedge=fell_on_wedge)
        # if fell_on_wedge[0]:
        #     self.task_env.execute_action(self.gym, self.sim, self.env, self.viewer, self.args, self.config, action)
        ############################################################################################################

        if len(actions) == 0:
            return self.task_env.execute_action_pinn(self.config, goal_pos, action, render=self.args.vis)
        if self.args.adaptive:
            # beta = 0
            mean_pred, std_pred = self.gaussian_process.predict([action], return_std=True)
            return self.task_env.execute_action_pinn(self.config, goal_pos, action, render=self.args.vis) + mean_pred[0] + beta * std_pred[0]
        else:
            return self.task_env.execute_action_pinn(self.config, goal_pos, action, render=self.args.vis)

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
            node.reward_history.append([reward])
            if self.args.vis:
                x = node
                while (x.depth > 0):
                    x = x.parent
                self.visualize_tree(x)

    def MCTS(self, node, goal_pos, actions, rewards, root = None):
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
            if root == None:
                reward = self.MCTS(child_node, goal_pos, actions, rewards, node)
            else:
                reward = self.MCTS(child_node, goal_pos, actions, rewards, root)
            node.reward_history[max_idx].append(reward)
            if self.use_softmax:
                node.val_children[max_idx] = self.softmax_value(node.reward_history[max_idx])
            else:
                node.val_children[max_idx] = (
                    node.val_children[max_idx] * node.num_actions[max_idx] +
                    reward) / (node.num_actions[max_idx] + 1)
            node.num_actions[max_idx] += 1
        if self.args.vis:
            while (node.depth > 0):
                node = node.parent
            self.visualize_tree(node)
        # if root == None and self.args.vis:
        #     self.visualize_tree(node)
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

    def search(self, goal_pos, execute_action=True):
        opt_reward = 1.0
        actions = []
        rewards = []
        best_reward = -math.inf
        for attempt in range(self.NUM_ACTIONS):
            if self.args.vis:
                update_label("ChosAct", "Chosen Action: ____")
                update_label("PredRew", "Pred. Reward: ____")
                update_label("IdealRew", "Ideal Reward: ____")
            if execute_action and self.args.vis:
                    update_label("Trial", "Trial: " + str(attempt + 1) + "/" + str(self.NUM_ACTIONS))

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
            if self.args.vis:
                self.visualize_tree(root_node)
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

                if abs(max_confidence - prev_conf) < 1.0e-4:
                    break

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

            print("Best PINN Action:", best_action)
            print("Best PINN Reward:", max_reward)
            if self.args.vis:
                update_label("ChosAct", "Chosen Action: " + str([round(i, 2) for i in best_action]))
                update_label("PredRew", "Pred. Reward: " + str(round(max_reward, 2)))
                time.sleep(1.0)
            # self.task_env.execute_action_pinn(self.config, goal_pos, best_action, render=True)
            # input("Simulated best action!")
            if (execute_action):
                
                reward_ideal = self.task_env.execute_action(self.gym, self.sim, self.env, self.viewer, self.args, self.config, best_action)
                reward_curr = self.task_env.execute_action_pinn(self.config, goal_pos, best_action, render=False)
                # input("Simulated best action!")

                print("IDEAL Reward:", reward_ideal)
                if self.args.vis:
                    update_label("IdealRew", "Ideal Reward: " + str(round(reward_ideal, 2)))
                    time.sleep(1.0)
                # print('Best Action:', best_action)
                # print('Reward:', reward_ideal)
                if reward_ideal > best_reward:
                    best_reward = reward_ideal
                actions.append(best_action)
                rewards.append(reward_ideal - reward_curr)

                # if max_reward < 1 and max_confidence < 1 and max_confidence > 0:
                if len(self.variable_list) == 0:
                    self.variable_list.append([])
                    self.variable_list.append([])
                    self.variable_list.append([])
                self.variable_list[1].append([max_confidence, max_reward])
                self.variable_list[2].append([max_confidence, reward_ideal])
            else:
                reward_curr = self.task_env.execute_action_pinn(self.config, goal_pos, best_action, render=self.args.vis)
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
