!/bin/bash

ENVIRONMENT="pendulum"
ACTION_PARAMETERS=2
CONTEXTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

for i in 1 2 3 4 5
do
    python llm_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=True
done

# # ******************************************************************************

ENVIRONMENT="sliding"
ACTION_PARAMETERS=2
CONTEXTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

for i in 1 2 3 4 5
do
    python llm_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=True
done

# ******************************************************************************

ENVIRONMENT="wedge"
ACTION_PARAMETERS=2
CONTEXTS=10

# rm experiments/regret_result_$ENVIRONMENT.txt

for i in 1 2 3 4 5
do
    python llm_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=True
done

# ******************************************************************************

ENVIRONMENT="sliding_bridge"
ACTION_PARAMETERS=3
CONTEXTS=1

rm experiments/regret_result_$ENVIRONMENT.txt

for i in 1 2 4
do
    python llm_eval_agent.py --env=$ENVIRONMENT --action_params=$ACTION_PARAMETERS --contexts=$CONTEXTS --actions=$i --simulate=True --robot=True
done
