#!/bin/bash

# ENVIRONMENT="pendulum"
# ACTION_PARAMETERS=2
# CONTEXTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

# for i in 1 2 3 4 5
# do
#     python random_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=true
# done

# # MODEL="model_wedge_actual_50.pth"
# # SIMS=1000
# # for i in 1 2 3 4 5
# # do
# #     python eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
# # done

# MODEL="model_pendulum_actual_100.pth"
# SIMS=2000
# for i in 1 2 3 4 5
# do
#     python eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS --simulate=true
# done

# # MODEL="model_wedge_actual_50.pth"
# # SIMS=1000
# # for i in 1 2 3 4 5
# # do
# #     python phyre_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
# # done

# MODEL="model_pendulum_actual_100.pth"
# SIMS=2000
# for i in 1 2 3 4 5
# do
#     python phyre_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS --simulate=true
# done

# # for i in 1 2 3 4 5
# # do
# #     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=true
# # done

# # python plot.py $ENVIRONMENT

# # ******************************************************************************

# ENVIRONMENT="sliding"
# ACTION_PARAMETERS=2
# CONTEXTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

# for i in 1 2 3 4 5
# do
#     python random_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=true
# done

# # MODEL="model_wedge_actual_50.pth"
# # SIMS=1000
# # for i in 1 2 3 4 5
# # do
# #     python eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
# # done

# MODEL="model_sliding_actual_100.pth"
# SIMS=2000
# for i in 1 2 3 4 5
# do
#     python eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS --simulate=true
# done

# # MODEL="model_wedge_actual_50.pth"
# # SIMS=1000
# # for i in 1 2 3 4 5
# # do
# #     python phyre_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
# # done

# MODEL="model_sliding_actual_100.pth"
# SIMS=2000
# for i in 1 2 3 4 5
# do
#     python phyre_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS --simulate=true
# done

# # for i in 1 2 3 4 5
# # do
# #     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=true
# # done

# # for i in 1 2 3 4 5
# # do
# #     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=true --perception=True
# # done

# python plot.py $ENVIRONMENT

# ******************************************************************************

# ENVIRONMENT="wedge"
# ACTION_PARAMETERS=2
# CONTEXTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

# for i in 1 2 3 4 5
# do
#     python random_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=true
# done

# MODEL="model_wedge_actual_50.pth"
# SIMS=1000
# for i in 1 2 3 4 5
# do
#     python eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
# done

# MODEL="model_wedge_actual_100.pth"
# SIMS=2000
# for i in 1 2 3 4 5
# do
#     python eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS --simulate=true
# done

# MODEL="model_wedge_actual_50.pth"
# SIMS=1000
# for i in 1 2 3 4 5
# do
#     python phyre_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
# done

# MODEL="model_wedge_actual_100.pth"
# SIMS=2000
# for i in 1 2 3 4 5
# do
#     python phyre_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS --simulate=true
# done

# for i in 1 2 3 4 5
# do
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=true
# done

# ******************************************************************************

ENVIRONMENT="sliding_bridge"
ACTION_PARAMETERS=3
CONTEXTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

for i in 1 2 3 4 5
do
    python random_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i
done

# MODEL="model_wedge_actual_50.pth"
# SIMS=1000
# for i in 1 2 3 4 5
# do
#     python eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
# done

MODEL="model_sliding_bridge_actual_100.pth"
SIMS=2000
for i in 1 2 3 4 5
do
    python eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
done

# MODEL="model_wedge_actual_50.pth"
# SIMS=1000
# for i in 1 2 3 4 5
# do
#     python phyre_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
# done

MODEL="model_sliding_bridge_actual_100.pth"
SIMS=2000
for i in 1 2 3 4 5
do
    python phyre_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --model=$MODEL --sims=$SIMS
done

# for i in 1 2 3 4 5
# do
#     python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i
# done

# python plot.py $ENVIRONMENT