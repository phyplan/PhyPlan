#!/bin/bash
L=800 # No of contexts
M=20 # No of actions/trials per context
L_TEST=200 # No of contexts
M_TEST=10 # No of actions/trials per context

rm experiments/time_result_$ENVIRONMENT.txt

# ********************************************************************************

# ENVIRONMENT=$1

# echo "Generating Train Data For" $ENVIRONMENT "Environment"
# python data_generation.py --env=$ENVIRONMENT --mode=train --L=$L --M=$M

# echo "Generating Test Data For" $ENVIRONMENT "Environment"
# python data_generation.py --env=$ENVIRONMENT --mode=test --L=$L_TEST --M=$M_TEST

# ********************************************************************************

ENVIRONMENT="wedge"

echo "Generating Train Data For" $ENVIRONMENT "Environment"
python data_generation.py --env=$ENVIRONMENT --mode=train --L=$L --M=$M

echo "Generating Test Data For" $ENVIRONMENT "Environment"
python data_generation.py --env=$ENVIRONMENT --mode=test --L=$L_TEST --M=$M_TEST

# --------------------------------------------------------------------------------

ENVIRONMENT="pendulum"

echo "Generating Train Data For" $ENVIRONMENT "Environment"
python data_generation.py --env=$ENVIRONMENT --mode=train --L=$L --M=$M

echo "Generating Test Data For" $ENVIRONMENT "Environment"
python data_generation.py --env=$ENVIRONMENT --mode=test --L=$L_TEST --M=$M_TEST

# --------------------------------------------------------------------------------

ENVIRONMENT="sliding"

echo "Generating Train Data For" $ENVIRONMENT "Environment"
python data_generation.py --env=$ENVIRONMENT --mode=train --L=$L --M=$M

echo "Generating Test Data For" $ENVIRONMENT "Environment"
python data_generation.py --env=$ENVIRONMENT --mode=test --L=$L_TEST --M=$M_TEST

# --------------------------------------------------------------------------------

ENVIRONMENT="sliding_bridge"

echo "Generating Train Data For" $ENVIRONMENT "Environment"
python data_generation.py --env=$ENVIRONMENT --mode=train --L=$L --M=$M

echo "Generating Test Data For" $ENVIRONMENT "Environment"
python data_generation.py --env=$ENVIRONMENT --mode=test --L=$L_TEST --M=$M_TEST

# ********************************************************************************
