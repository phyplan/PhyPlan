#!/bin/sh

# ENVIRONMENT="wedge"
# ACTION_PARAMETERS=2
# CONTEXTS=10
# PINN_ATTEMPTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

# for i in 1 2 3 4 5
# do
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --simulate=true
# done

# for i in 1 2 3 4 5
# do
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --simulate=true --perception=True
# done

# ******************************************************************************

# ENVIRONMENT="pendulum"
# ACTION_PARAMETERS=2
# CONTEXTS=10
# PINN_ATTEMPTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

# for i in 1 2 3 4 5
# do
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --simulate=true
# done

# for i in 1 2 3 4 5
# do
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --perception=True
# done

# ******************************************************************************

# ENVIRONMENT="sliding"
# ACTION_PARAMETERS=2
# CONTEXTS=10
# PINN_ATTEMPTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

# for i in 1 2 3 4 5
# do
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --simulate=true
# done

# for i in 1 2 3 4 5
# do
#     # python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --perception=True --simulate=true
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --perception=True
# done

# ******************************************************************************

# ENVIRONMENT="sliding_bridge"
# ACTION_PARAMETERS=3
# CONTEXTS=10
# PINN_ATTEMPTS=20

# rm experiments/regret_result_$ENVIRONMENT.txt

# for i in 1 2 3 4 5
# do
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS
# done

# for i in 1 2 3 4 5
# do
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --perception=True
# done

# ==============================================================================
# ##############################################################################
# ==============================================================================

ACTION_PARAMETERS=2
PINN_ATTEMPTS=10
CONTEXTS=1
ACTIONS=50

# ENVIRONMENT="pendulum"

# python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --pinn_attempts=$PINN_ATTEMPTS --perception=True
# python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --pinn_attempts=$PINN_ATTEMPTS --perception=True --adaptive=False

# ******************************************************************************

ENVIRONMENT="wedge"

# python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --pinn_attempts=$PINN_ATTEMPTS --perception=True
# python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --pinn_attempts=$PINN_ATTEMPTS --perception=True --adaptive=False

python random_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --simulate=True

MODEL="model_wedge_actual_100.pth"
SIMS=2000
python eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --model=$MODEL --sims=$SIMS --simulate=True

MODEL="model_wedge_actual_100.pth"
SIMS=2000
python phyre_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --model=$MODEL --sims=$SIMS --simulate=True

# ******************************************************************************

# ENVIRONMENT="sliding"

# python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --pinn_attempts=$PINN_ATTEMPTS --perception=True
# python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --pinn_attempts=$PINN_ATTEMPTS --perception=True --adaptive=False

# ******************************************************************************

# ENVIRONMENT="sliding_bridge"
# ACTION_PARAMETERS=3
# PINN_ATTEMPTS=20

# python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --pinn_attempts=$PINN_ATTEMPTS --perception=True
# python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$ACTIONS --pinn_attempts=$PINN_ATTEMPTS --perception=True --adaptive=False
