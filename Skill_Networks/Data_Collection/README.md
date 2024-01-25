# Data-Collecion
Code for data collection for Skill Learning Networks

We have created two pipelines for Data Collection: 1) Without Perception (or directly from simulator) and  2) With Perception (Image based Data Collection)

## Installation
Follow the installation process for [Isaac Gym](https://developer.nvidia.com/isaac-gym). We recommend setting up a dedicated conda environment. No other installation is required.

## Activating Environment
Run commands in `setup.sh`

## Generating Training Data
Run `python data_generation.py` with specific arguments to generate the data. Various arguments are:

| Skill | Description |
| --- | --- |
| Bouncing | A ball bouncing on wedge |
| Hitting  | A pendulum hitting a puck |
| Throwing | A ball performing projectile motion |
| Swinging | A pendulum swinging |
| Sliding | A box sliding on fixed surface |


| Skill | Data Collected |
| --- | --- |
| `Bouncing` | Simulation Number, Data Number, Time elapsed, Correlation coefficient, Wedge  Angle, Velocity of ball in x direction before collision, Velocity of ball in y direction before collision, Velocity of ball in x direction after collision, Velocity of ball in y direction after collision  |
| `Hitting`  | Simulation Number, Pendulum Bob mass, Puck/Peg Mass, Velocity of Pendulum Bob before hitting, Velocity of Peg after hitting|
| `Throwing` | Episode Number, Data Number in that Episode, Time Elapsed, Initial Velocity in Y direction, Initial Velocity in X direction, Current Velocity in Y direction, Displacement in Y direction, Displacement in X direction |
| `Swinging` | Episode Number, Data Number in that Episode, Initial Theta, Time Elapsed, Current Theta, Current omega |
| `Sliding` | Episode Number, Data Number in that Episode, Actual Mu, Initial Velocity, Time Elapsed, Current Velocity, Distance Covered |

Note: Each of these skills have two separate mode for data generation: with and without Perception.

## Run 
Run `python *_main.py` along with desired configurations in respective config for data generation.