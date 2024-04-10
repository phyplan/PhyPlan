import ast
import time
import requests
from flask import Flask, request, jsonify
import google.generativeai as genai

GOOGLE_API_KEY='AIzaSyDAe8RHMBAMwqvy8Wzp1q8zWoTwe3K8P0M'

genai.configure(api_key=GOOGLE_API_KEY, transport='rest')

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
print('--------------')


def generate_first_prompt(env, goal_pos):
    if env == 'wedge':
        return f"There is a robot and a goal located at {goal_pos} outside the direct reach of the robot. There is a ball that needs to reach the goal. The environment has a wedge (an inclined plane at 45 degrees from the horizontal plane) placed at origin, and the robot can bounce the ball over the wedge to place the ball inside the goal. The height of the wedge centre from the ground is fixed at 0.3 metres. The robot can orient the wedge along any horizontal direction and choose to drop the ball over the wedge from any height. When dropped from a height, the ball bounces on the wedge and lands far away on the ground. \
        Sanity check 1: How does the orientation angle of the wedge affect the ball's position with respect to the goal? \
        Sanity check 2: How does the drop height of the ball affect the ball's position with respect to the goal?"
    elif env == 'sliding':
        return f"There is a robot and a goal located at {goal_pos} outside the direct reach of the robot. There is a puck that needs to reach the goal. The environment has a fixed table over which the puck slides, and a pendulum hanging over the puck that the robot can orient to hit the puck to slide it to the goal. The robot can orient the pendulum along any vertical plane and choose to drop the pendulum from any angle from the vertical axis. When hit with a pendulum, the puck slides on the table. \
        Sanity check 1: How does the plane of the pendulum affect the puck's position with respect to the goal? \
        Sanity check 2: How does the drop angle of the pendulum affect the puck's position with respect to the goal?"
    elif env == 'pendulum':
        return f"There is a robot and a goal located at {goal_pos} outside the direct reach of the robot. There is a ball that needs to reach the goal. The environment has a fixed pillar over which the ball is resting, and a pendulum hanging over the ball that the robot can orient to hit the ball to throw it to the goal. The robot can orient the pendulum along any vertical plane and choose to drop the pendulum from any angle from the vertical axis. When hit with a pendulum, the puck projectiles and lands far away on the ground. \
        Sanity check 1: How does the plane of the pendulum affect the puck's position with respect to the goal? \
        Sanity check 2: How does the drop angle of the pendulum affect the puck's position with respect to the goal?"
    elif env == 'sliding_bridge':
        return f"There is a robot and a goal located at {goal_pos} outside the direct reach of the robot. There is a puck that needs to reach the goal. The environment has a fixed table over which the puck slides, a movable bridge over which the puck slides and a pendulum that the robot can orient to move the puck towards the goal. The robot can orient the pendulum along any vertical plane, orient the bridge in any horizontal direction and choose to drop the pendulum from any angle from the vertical axis. When hit with a pendulum, the puck slides on the table, then on the bridge and finally projectiles to land far away on the ground. \
        Sanity check 1: How does the plane of the pendulum affect the puck's position with respect to the goal? \
        Sanity check 2: How does the drop angle of the pendulum affect the puck's position with respect to the goal? \
        Sanity check 3: How does the orientation angle of the bridge affect the puck's position with respect to the goal?"
    else:
        return "Sorry! Unknown Environment."


def generate_second_prompt(env, bnds):
    if env == 'wedge':
        return f"In one line, give the numerical values of the angle to orient the wedge and the height to drop the ball from in the format (angle in decimal radians, height in meters). The bound for angle is ({bnds[0][0]}, {bnds[0][1]}) and that for height is ({bnds[1][0]}, {bnds[1][1]}). I will tell you where the ball landed, and you should modify your answer accordingly till the ball reaches the goal. I have marked the ground into two halves. The goal lies in one half, and the robot and the wedge are at the centre. Thoughout the conversation, remember that my response would be one of these: \n \
            1. The ball lands in the half not containing goal, I'd say 'WRONG HALF'. \
            2. The ball lands in the correct half but left of the goal, I'd say 'LEFT by <horizontal distance between ball and goal>'. \
            3. The ball lands in the correct half but right of the goal, I'd say 'RIGHT by <horizontal distance between ball and goal>'. \
            4. The ball lands in the correct half and in line but overshot the goal, I'd say 'OVERSHOT by <horizontal distance between ball and goal>'. \
            5. The ball lands in the correct half and in line but fell short of the goal, I'd say 'FELL SHORT by <horizontal distance between ball and goal>'. \
            6. Finally, the ball successfully landed in the goal, I'd say 'GOAL'. \
        Note: In your response, do not write anything else except the (angle, height) pair. Send in tuple FORMAT: (angle, height). Do not emphasise the answer, just return plain text. Let's begin with an initial guess!"
    elif env == 'sliding':
        return f"In one line, give the numerical values of the angle to orient the pendulum's plane and the angle to drop the pendulum from (both in decimal radians). The bound for plane orientation angle is ({bnds[0][0]}, {bnds[0][1]}) and that for drop angle with vertical axis is ({bnds[1][0]}, {bnds[1][1]}). I will tell you where the puck landed, and you should modify your answer accordingly till the puck reaches the goal. I have marked the ground into two halves. The goal lies in one half, and the robot and the wedge are at the centre. Thoughout the conversation, remember that my response would be one of these: \n \
            1. The puck lands in the half not containing goal, I'd say 'WRONG HALF'. \
            2. The puck lands in the correct half but left of the goal, I'd say 'LEFT by <horizontal distance between puck and goal>'. \
            3. The puck lands in the correct half but right of the goal, I'd say 'RIGHT by <horizontal distance between puck and goal>'. \
            4. The puck lands in the correct half and in line but overshot the goal, I'd say 'OVERSHOT by <horizontal distance between puck and goal>'. \
            5. The puck lands in the correct half and in line but fell short of the goal, I'd say 'FELL SHORT by <horizontal distance between puck and goal>'. \
            6. Finally, the puck successfully landed in the goal, I'd say 'GOAL'. \
        Note: In your response, do not write anything else except the (pendulum's plane angle, pendulum's drop angle) pair. Send in tuple FORMAT: (angle 1, angle 2). Do not emphasise the answer, just return plain text. Let's begin with an initial guess!"
    elif env == 'pendulum':
        return f"In one line, give the numerical values of the angle to orient the pendulum's plane and the angle to drop the pendulum from (both in decimal radians). The bound for plane orientation angle is ({bnds[0][0]}, {bnds[0][1]}) and that for drop angle with vertical axis is ({bnds[1][0]}, {bnds[1][1]}). I will tell you where the ball landed, and you should modify your answer accordingly till the ball reaches the goal. I have marked the ground into two halves. The goal lies in one half, and the robot and the wedge are at the centre. Thoughout the conversation, remember that my response would be one of these: \n \
            1. The ball lands in the half not containing goal, I'd say 'WRONG HALF'. \
            2. The ball lands in the correct half but left of the goal, I'd say 'LEFT by <horizontal distance between ball and goal>'. \
            3. The ball lands in the correct half but right of the goal, I'd say 'RIGHT by <horizontal distance between ball and goal>'. \
            4. The ball lands in the correct half and in line but overshot the goal, I'd say 'OVERSHOT by <horizontal distance between ball and goal>'. \
            5. The ball lands in the correct half and in line but fell short of the goal, I'd say 'FELL SHORT by <horizontal distance between ball and goal>'. \
            6. Finally, the ball successfully landed in the goal, I'd say 'GOAL'. \
        Note: In your response, do not write anything else except the (pendulum's plane angle, pendulum's drop angle) pair. Send in tuple FORMAT: (angle 1, angle 2). Do not emphasise the answer, just return plain text. Let's begin with an initial guess!"
    elif env == 'sliding_bridge':
        return f"In one line, give the numerical values of the angle to orient the pendulum's plane, the angle to orient the bridge and the angle to drop the pendulum from (all in decimal radians). The bound for plane orientation angle is ({bnds[0][0]}, {bnds[0][1]}), that for bridge orientation angle is ({bnds[2][0]}, {bnds[2][1]}), and that for drop angle with vertical axis is ({bnds[1][0]}, {bnds[1][1]}). I will tell you where the puck landed, and you should modify your answer accordingly till the puck reaches the goal. I have marked the ground into two halves. The goal lies in one half, and the robot and the wedge are at the centre. Thoughout the conversation, remember that my response would be one of these: \n \
            1. The puck lands in the half not containing goal, I'd say 'WRONG HALF'. \
            2. The puck lands in the correct half but left of the goal, I'd say 'LEFT by <horizontal distance between puck and goal>'. \
            3. The puck lands in the correct half but right of the goal, I'd say 'RIGHT by <horizontal distance between puck and goal>'. \
            4. The puck lands in the correct half and in line but overshot the goal, I'd say 'OVERSHOT by <horizontal distance between puck and goal>'. \
            5. The puck lands in the correct half and in line but fell short of the goal, I'd say 'FELL SHORT by <horizontal distance between puck and goal>'. \
            6. Finally, the puck successfully landed in the goal, I'd say 'GOAL'. \
        Note: In your response, do not write anything else except the (pendulum's plane angle, pendulum's drop angle, bridge's orientation angle) triplet. Send in tuple FORMAT: (angle 1, angle 2, angle 3). Do not emphasise the answer, just return plain text. Let's begin with an initial guess!"
    else:
        return "Sorry! Unknown Environment."


model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
env = ''
bnds = None
num_contexts = 0
num_actions = 1

# ------------------------------------------ #
def get():
    global env, bnds, chat, num_actions, num_contexts
    response = request.get_json()
    print('--------------- RESPONSE: --', response)
    
    iter = response['iter']
    if iter == 0:
        env = response['env']
        bnds = response['bnds']
        num_actions = int(response['num_actions'])
        num_contexts = int(response['num_contexts'])

    if iter % num_actions == 0:
        goal_pos = response['goal_pos']
        chat = model.start_chat(history=[])

        response = chat.send_message(generate_first_prompt(env, goal_pos))
        print(response.text, '\n--------------')

        response = chat.send_message(generate_second_prompt(env, bnds))
        print('Action Generated:', response.text, '\n--------------')
        
        action = response.text.split('\n')[0].replace('*', '')
        action = action.replace('`', '')
        action = action.replace('\'', '')
        action = action.replace('\"', '')
        data = {
            'action': action
        }
        return jsonify(data)
    else:
        feedback = response['message']

        print('Feedback Received:', feedback)
        response = chat.send_message(feedback)
        print('Action Generated:', response.text, '\n--------------')

        action = response.text.split('\n')[0].replace('*', '')
        action = action.replace('`', '')
        action = action.replace('\'', '')
        action = action.replace('\"', '')
        data = {
            'action': action
        }
        return jsonify(data)


app = Flask(__name__)
app.route('/send', methods=['POST'])(get)
app.run(host='0.0.0.0', port=7002)
