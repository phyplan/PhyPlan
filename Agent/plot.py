import matplotlib.pyplot as plt
import numpy as np
import sys

env_name = {'pendulum': 'Launch', 'sliding': 'Slide', 'wedge': 'Bounce', 'sliding_bridge': 'Bridge', 'paddles': 'Paddles'}

env = sys.argv[1]
file_name = "experiments/regret_result_%s_all.txt" % (env)

# title = sys.argv[3]
# name = sys.argv[2]
# file_name = (f'experiments/{sys.argv[1]}')

# plt.rcParams['font.size'] = '15'

data = {}
agents = []
agent = ""

with open(file_name, 'r') as f:
    for line in f.readlines():
        words = line.split()
        if len(words) == 1:
            agent = words[0]
            if agent not in agents:
                agents.append(agent)
        elif len(words) > 1:
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

markers = ['o', 'd', 's', '|', '^', '*', 'h', '<', 'H', '+', '>', 'x', 'D', 'd', '|', '_']

fig, ax = plt.subplots(figsize=(7, 6))

i = 0
for agent in agents:
    actions = data[agent + '_actions']
    regret = data[agent + '_regret']
    std = data[agent + '_std']
    ax.plot(actions, regret, label=agent.replace('_', ' '), marker=markers[i])
    ax.fill_between(actions, (regret - std), (regret + std), alpha=.2)
    i = (i + 1) % len(markers)

plt.ylabel('Regret', fontsize='20')
plt.ylim(0, None)
plt.xlabel('Actions', fontsize='20')
plt.title(env_name[env], fontsize='30')
plt.xticks([1, 2, 3, 4, 5], fontsize='20')
plt.yticks(fontsize='20')
# plt.title(title)
plt.legend(fontsize='15')
plt.savefig('plots/' + env + '.png', bbox_inches='tight')
# plt.show()

fig.clear()