import matplotlib.pyplot as plt
import numpy as np
import sys

env_name = {'pendulum': 'Launch', 'sliding': 'Slide', 'wedge': 'Bounce', 'sliding_bridge': 'Bridge', 'paddles': 'Paddles'}

env = sys.argv[1]
file_name = "experiments/position_%s.txt" % (env)

# title = sys.argv[3]
# name = sys.argv[2]
# file_name = (f'experiments/{sys.argv[1]}')

data = {}
agents = []
agent = ""
iteration = -1

with open(file_name, 'r') as f:
    for line in f.readlines():
        words = line.split()
        print(words)
        if len(words) == 1:
            agent = words[0]
            if agent not in agents:
                agents.append(agent)
        elif len(words) > 1:
            if 'Iteration' in words:
                iteration = int(words[1])
            elif 'Goal_Pos:' in words:
                item = agent + '-goal_pos'
                data[item] = np.array([float(words[1][1:-1]), float(words[2][:-1])])
            else:
                item1 = agent + '-ball_pos_X'
                item2 = agent + '-ball_pos_Y'
                if item1 not in data:
                    data[item1] = np.array([])
                if item2 not in data:
                    data[item2] = np.array([])
                data[item1] = np.append(data[item1], float(words[0][1:-1]))
                data[item2] = np.append(data[item2], float(words[1][:-1]))

print(data)

markers = ['o', '_', 's', '|', '^', '*', 'h', '<', 'H', '+', '>', 'x', 'D', 'd', '|', '_']

def create_spectrum(width):
    colors = []
    ranges = np.linspace(0, 6, width)
    ranges = np.round(ranges * 1000) / 1000
    step = np.round((ranges[1] - ranges[0]) * 1000) / 1000
    
    red = 1
    green = 0
    blue = 0
    for i in ranges:
        if i > 0.000 and i <= 1.000:
            blue += step
        elif i > 1.000 and i <= 2.000:
            red -= step
        elif i > 2.000 and i <= 3.000:
            green += step
        elif i > 3.000 and i <= 4.000:
            blue -= step
        elif i > 4.000 and i <= 5.000:
            red += step
        elif i > 5.000 and i <= 6.000:
            green -= step

        red = np.round(red * 1000) / 1000
        green = np.round(green * 1000) / 1000
        blue = np.round(blue * 1000) / 1000
        if red < 0:
            red = 0
        if red > 1:
            red = 1
        if green < 0:
            green = 0
        if green > 1:
            green = 1
        if blue < 0:
            blue = 0
        if blue > 1:
            blue = 1
        colors.append((red, green, blue))
    
    return colors

for agent in agents:
    fig, ax = plt.subplots()
    goal_pos = data[agent + '-goal_pos']
    ball_pos_X = data[agent + '-ball_pos_X']
    ball_pos_Y = data[agent + '-ball_pos_Y']
    # ax.plot(ball_pos_X, ball_pos_Y)
    # ax.scatter([goal_pos[0]], [goal_pos[1]], color='red', marker=',')
    ax.plot(goal_pos[0], goal_pos[1], color='red', marker='s')
    ax.annotate('GOAL', xy=(goal_pos[0], goal_pos[1]), xytext=(goal_pos[0] - 0.1, goal_pos[1] + 0.4), arrowprops=dict(arrowstyle='->'), fontsize='15')
    # ax.scatter(ball_pos_X, ball_pos_Y, color='green', marker='x')
    
    # colours = create_spectrum(len(ball_pos_X))
    # ax.scatter(ball_pos_X, ball_pos_Y, c=colours, marker='x')

    # iter = [f'Iteration {i}' for i in range(1, len(ball_pos_X) + 1)]
    iter = [i for i in range(0, len(ball_pos_X))]
    cs = ax.scatter(ball_pos_X, ball_pos_Y, c=iter, cmap='viridis', marker='x')
    cb = fig.colorbar(cs, ax=ax, location='right')
    cb.ax.set_title('Actions', fontsize='15')
    cb.ax.tick_params(labelsize=15)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=7)

    plt.ylabel('Y', fontsize='15', rotation=0)
    # plt.ylim(-1.25, 0.25)
    # plt.xlim(-0.5, 1.75)
    
    # plt.ylim(-0.8, -0.1)
    # plt.xlim(0.1, 0.9)
    
    plt.ylim(-1.5, 1.25)
    plt.xlim(-1.5, 1.75)
    
    plt.xlabel('X', fontsize='15')
    plt.xticks(fontsize='15')
    plt.yticks(fontsize='15')
    plt.title(f'{agent} ({env_name[env]})'.replace('_', ' '), fontsize='15')
    plt.savefig('plots/heat_map_' + agent + '_' + env + '.png', bbox_inches='tight')
    
    fig.clear()

# plt.show()

# fig.clear()