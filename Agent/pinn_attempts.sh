#!/bin/sh

ENVIRONMENT="sliding_bridge"
ACTION_PARAMETERS=3
CONTEXTS=5
NUM_ACTIONS=1

rm experiments/regret_result_$ENVIRONMENT.txt

PINN_ATTEMPTS=5
python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$NUM_ACTIONS --pinn_attempts=$PINN_ATTEMPTS --simulate=true --perception=True

PINN_ATTEMPTS=10
python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$NUM_ACTIONS --pinn_attempts=$PINN_ATTEMPTS --simulate=true --perception=True

PINN_ATTEMPTS=20
python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$NUM_ACTIONS --pinn_attempts=$PINN_ATTEMPTS --simulate=true --perception=True

PINN_ATTEMPTS=30
python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$NUM_ACTIONS --pinn_attempts=$PINN_ATTEMPTS --simulate=true --perception=True

PINN_ATTEMPTS=50
python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$NUM_ACTIONS --pinn_attempts=$PINN_ATTEMPTS --simulate=true --perception=True

python3 plot.py $ENVIRONMENT