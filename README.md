# Deep Reinforcement Learning - DDPG Network (Critic / Actor Network) - Unity Continuous Control Project

## Project Environment Description

This project is attempting to solve the problem presented in the [Udacity](https://www.udacity.com/) Deep Reinforcement Learning (DRL) course for Artificial Intelligence that asks to set up a policy based approach to  train the computer to maximize the score provided by the environment.  

The problem contains an environment in which there are two opposing tennis rackets, in which each racket can move on an axis parallel to the table as well as an axis perpendicular to the table (for two degrees of freedom each).  
The objective is for the rackets to keep a ball in the air as much as possible without letting it hit the table or leave the environment space.
To score this scenario, we use the maximum score between the two rackets for a given episode.  A score of 0.1 is given each time a racket hits a ball.
If the ball hits the table or leaves the allowed environment space, a score is given of -0.01.
The scenario is solved with this score achieves and average of 0.5 for a given racket is obtained over 100 episodes.

## Learning Environment Set Up

General instructions are available on the [Udacity github page for this project](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

This implementation was validated on a x64 based Windows system using Anaconda3 to provide the hosted environment environment.

1. Setup the conda environment
```
conda create --name collab-compet python=3.6 
activate collab-compet
```
2. Install dependecies
```
# Install general dependencies for Udacity deep-reinforcement-learning projects
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .

# Intall google OpenAI gym dependencies
pip install gym

# Install pytorch 0.4.1
conda install pytorch=0.4.1 cuda92 -c pytorch

# Install the old version of Unity ml-agents
pip install unityagents

# Install the ipython kernel package
conda install ipykernel
# Install the ipython kernel to the conda environment
python -m ipykernel install --user --name collab-compet --display-name "collab-compet"
```

To run the notebook, within the collab-compet conda environment run `jupyter lab` and navigate to the the appropriate ipython notebook.


## Using this program

Learning and a resultant view of both a random state, and learned state are achievable by following the progression in the Navigation-checkpoint.ipynb python notebook.
