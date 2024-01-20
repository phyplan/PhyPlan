import math

actual_file = open('data/actions_dryrun_sliding_bridge_actual.txt', 'r')
test_file = open('data/actions_dryrun_sliding_bridge_pinn.txt', 'r')

actual_lines = actual_file.readlines()
test_lines = test_file.readlines()

loss = 0.0
for i in range(len(actual_lines)):
    reward_actual = float(actual_lines[i].split()[-1])
    reward_test = float(test_lines[i].split()[-1])
    loss += (reward_actual - reward_test) ** 2
loss = loss / len(actual_lines)
loss = math.sqrt(loss)
print('Loss:', loss)