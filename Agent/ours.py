# Ours
# Action Parameters: 2.000000, Contexts: 4.000000, Actions: 2.000000, D: 20.000000
# Ours
# Action Parameters: 2.000000, Contexts: 4.000000, Actions: 1.000000, D: 20.000000
# Ours
# Action Parameters: 2.000000, Contexts: 4.000000, Actions: 2.000000, D: 20.000000
# Ours
# Action Parameters: 2.000000, Contexts: 4.000000, Actions: 3.000000, D: 20.000000

import matplotlib.pyplot as plt
import numpy as np
import sys

env_name = {'pendulum': 'Launch', 'sliding': 'Slide', 'wedge': 'Bounce', 'sliding_bridge': 'Bridge', 'paddles': 'Paddles'}

env = "wedge"
 
data = {}
agents = ["Wedge Perception (SAM)","Wedge Actual"]
agent = ""

file_names = ["experiments/regret_result_wedge.txt","experiments/regret_result_wedge_without_perception.txt"]
data = {}
i = 0
for file_name in file_names:
    agent = agents[i]
    i+=1
    with open(file_name, 'r') as f:
        for line in f.readlines():
            words = line.split()
            # if len(words) == 1:
            #     agent = words[0]
            #     if agent not in agents:
            #         agents.append(agent)
            if len(words) > 1:
                if 'Actions:' in words:
                    item = agent + '_actions'
                    if item not in data:
                        data[item] = np.array([])
                    temp = words[words.index('Actions:') + 1]
                    if temp[-1] == ',':
                        temp = temp[:-1]
                    val = float(temp)
                    data[item] = np.append(data[item], val)
                elif 'Final' in words:
                    item1 = agent + '_regret'
                    if item1 not in data:
                        data[item1] = np.array([])
                    val1 = float(words[1])
                    data[item1] = np.append(data[item1], val1)

                    item2 = agent + '_std'
                    if item2 not in data:
                        data[item2] = np.array([])
                    val2 = float(words[2])
                    data[item2] = np.append(data[item2], val2)

print(data)

fig, ax = plt.subplots()
for agent in agents:
    actions = data[agent + '_actions']
    regret = data[agent + '_regret']
    std = data[agent + '_std']
    ax.plot(actions,regret, label=agent)
    ax.fill_between(actions, (regret - std), (regret + std), alpha=.2)

plt.ylabel('Regret')
plt.ylim(0, None)
plt.xlabel('Actions')
plt.title(env_name[env])
plt.legend()
plt.savefig('plots/' + env + '.png')
plt.show()

fig.clear()