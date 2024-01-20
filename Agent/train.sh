#!/bin/bash

# ENVIRONMENT="sliding_bridge"
# EPOCHS=10
# CONTEXTS=100
# CONTEXTS_STEP=50
# ACTION_PARAMETERS=3

# python train.py --env=$ENVIRONMENT --epochs=$EPOCHS --contexts=$CONTEXTS --contexts_step=$CONTEXTS_STEP --action_params=$ACTION_PARAMETERS
# python train.py --env=$ENVIRONMENT --epochs=$EPOCHS --contexts=$CONTEXTS --contexts_step=$CONTEXTS_STEP --action_params=$ACTION_PARAMETERS --fast=True

# ********************************************************************************

ENVIRONMENT="wedge"
EPOCHS=50
CONTEXTS=800
CONTEXTS_STEP=100
ACTION_PARAMETERS=2

python train.py --env=$ENVIRONMENT --epochs=$EPOCHS --contexts=$CONTEXTS --contexts_step=$CONTEXTS_STEP --action_params=$ACTION_PARAMETERS

# --------------------------------------------------------------------------------

ENVIRONMENT="pendulum"
EPOCHS=50
CONTEXTS=800
CONTEXTS_STEP=100
ACTION_PARAMETERS=2

python train.py --env=$ENVIRONMENT --epochs=$EPOCHS --contexts=$CONTEXTS --contexts_step=$CONTEXTS_STEP --action_params=$ACTION_PARAMETERS

# --------------------------------------------------------------------------------

ENVIRONMENT="sliding"
EPOCHS=50
CONTEXTS=800
CONTEXTS_STEP=100
ACTION_PARAMETERS=2

python train.py --env=$ENVIRONMENT --epochs=$EPOCHS --contexts=$CONTEXTS --contexts_step=$CONTEXTS_STEP --action_params=$ACTION_PARAMETERS

# --------------------------------------------------------------------------------

ENVIRONMENT="sliding_bridge"
EPOCHS=50
CONTEXTS=800
CONTEXTS_STEP=100
ACTION_PARAMETERS=3

python train.py --env=$ENVIRONMENT --epochs=$EPOCHS --contexts=$CONTEXTS --contexts_step=$CONTEXTS_STEP --action_params=$ACTION_PARAMETERS

# --------------------------------------------------------------------------------

