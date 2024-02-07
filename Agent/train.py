from isaacgym import gymapi, gymutil
import ast
import torch
from model import *
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

args = gymutil.parse_arguments(
    custom_parameters = [
        {"name": "--env", "type": str, "default": "pendulum", "help": "Specify Environment"},
        {"name": "--fast", "type": ast.literal_eval, "default": False, "help": "Invoke Fast Simulator"},
        {"name": "--epochs", "type": int, "default": 40, "help": "Number of Epochs"},
        {"name": "--contexts", "type": int, "default": 100, "help": "Number of Contexts"},
        {"name": "--contexts_step", "type": int, "default": 10, "help": "Number of Step Contexts"},
        {"name": "--action_params", "type": int, "default": 2, "help": "Number of Action Parameters"},
    ]
)

torch.manual_seed(0)

NUM_EPOCHS = args.epochs
NUM_CONTEXTS = args.contexts
NUM_CONTEXTS_STEP = args.contexts_step

ACTION_PARAMETERS = args.action_params

train_folder = 'data/images_train_' + args.env + ('_pinn' if args.fast else '_actual')
test_folder = 'data/images_test_' + args.env + ('_pinn' if args.fast else '_actual')
train_file = 'data/actions_train_' + args.env + ('_pinn' if args.fast else '_actual') + '.txt'
test_file = 'data/actions_test_' + args.env + ('_pinn' if args.fast else '_actual') + '.txt'

dataset = ImageDataset(train_folder, train_file, NUM_CONTEXTS)
test_dataset = ImageDataset(test_folder, test_file, NUM_CONTEXTS)
dataloader = DataLoader(dataset, batch_size = 1)
model = ResNet18FilmAction(ACTION_PARAMETERS, fusion_place = 'last_single').to(device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

def testLoss():
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            if i == len(test_dataset):
                break
            item = test_dataset[i]
            img = item['img'].to(device)
            action = item['action'].to(device)
            reward = item['reward'].to(device)
            output = model(img, action)
            loss = F.mse_loss(output, reward)
            running_loss += loss.item()
    return running_loss / len(test_dataset)

for num_contexts in range(NUM_CONTEXTS_STEP, NUM_CONTEXTS + 1, NUM_CONTEXTS_STEP):
    running_loss = 0.0
    count = 0
    least_test_loss = 1e9
    for epoch in range(NUM_EPOCHS):
        with tqdm(dataset) as tepoch:
            for idx, item in enumerate(tepoch):
                if idx > num_contexts or idx == len(dataset) - 1:
                    break
                model.train()
                img = item['img'].to(device)
                action = item['action'].to(device)
                reward = item['reward'].to(device)
                output = model(img, action)
                loss = F.mse_loss(output, reward)
                running_loss += loss.item()
                count += 1
                model.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=running_loss / count)
        print('Epoch: {}, Loss: {}'.format(epoch, running_loss/num_contexts))
        test_loss = testLoss()
        print('Test Loss: {}'.format(test_loss))
        running_loss = 0.0
        count = 0
        if test_loss < least_test_loss:
            save_path = 'agent_models/model_%s_%s_%d.pth' % (args.env, ('pinn' if args.fast else 'actual'), num_contexts)
            torch.save(model.state_dict(), save_path)
            least_test_loss = test_loss
            print('Model Saved!')
