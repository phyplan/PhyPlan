#!/bin/bash
ENVIRONMENT="wedge"
ACTION_PARAMETERS=2
CONTEXTS=10
PINN_ATTEMPTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

# ************ WITHOUT PERCEPTION ************

for i in 1 2 3 4 5
do
    python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS
done

for i in 1 2 3 4 5
do
    python pinn_mcts_eval_agent_kernel.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS
done

for i in 1 2 3 4 5
do
    python pinn_mcts_eval_agent_widening.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS
done

for i in 1 2 3 4 5
do
    python pinn_mcts_eval_agent_kernel_widening.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS
done

# ************** WITH PERCEPTION **************

for i in 1 2 3 4 5
do
    python pinn_mcts_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --perception=True
done

for i in 1 2 3 4 5
do
    python pinn_mcts_eval_agent_kernel.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --perception=True
done

for i in 1 2 3 4 5
do
    python pinn_mcts_eval_agent_widening.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --perception=True
done

for i in 1 2 3 4 5
do
    python pinn_mcts_eval_agent_kernel_widening.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --pinn_attempts=$PINN_ATTEMPTS --perception=True
done
