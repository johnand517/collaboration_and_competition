## Project Objectives and Setup

We will use a Multi-Agent Deep Deterministic Policy Gradient ([MADDPG](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)) 
network to train the agent to solve the problem, 
This approach uses an actor network that is shared for each agent and a critic network unique to each agent, and the goal
is to arrive at an optimal policy for each agent.
All networks are randomly initialized upon onset.  To achieve our goal, each agent estimates next best actions from the common
 actor network.  We then feed the next best actions for all agents into each agent's critic next, along with that critic's
state, to obtain Q value estimates, These Q-value estimates are then used to get a loss function for our actor network
which allows us to update the actor network.

To achieve this, we will require:
- A sequential deep neural network that estimates actions based on the provided state (actor network)
- A sequential deep neural network that estimates Q values given states + actions (critic network) for each agent
- For all of the above networks, we will keep a local and target network, in which the local network learns actively on each iteration, and the target network is updated more slowly through a soft update parameter.
- A function that adds noise to our selected actions
- A replay buffer that stores prior experiences as our agent learns from the environment.  A single experience contains
actions, states, next states, rewards, and whether an agent's episode has completed for all agents.

Our program acts upon the environment by initially choosing actions for all agents, for each of their given states more-or-less at random, 
and then for each agent determines the rewards, and next states given the actions chosen.  
This "experience" is then stored in the replay buffer.  We then update the current states to the next states determined from the chosen action and repeat this process.

Learning from the environment happens by choosing a sample of experiences from the replay buffer after a determined number of steps through the environment.  From these sampled experiences, 
we calculate successive actions for each agent given each agent's given the experience provided next agent states using our target actor network.  
To evaluate we iterate over all of the agent's critic networks by passing to each, all actions, and the corresponding agent state + next state + 
reward to the given agent's critic network.
For each critic network, we then compute a target Q value (using the target network), and an estimated Q value (using the local network) 
and use these two values to calculate a loss for our critic network.  This loss is then used to update the local critic network.

As part of updating our critic network, we using gradient clipping to reduce the magnitude of the norm of the vector to be less than or equal than 1, to prevent a gradient from getting too large
and causing unexpected divergence.

We then train our actor network by obtaining next best actions for each agent (local actor network), and calculate a loss function for our actor network 
by feeding these actions into our local critic network for a given agent.  This learning is done for every agent's critic network
update iteration using the given agent's critic network and current state. 

Our critic networks for each agent have a soft update with each iteration.  Our actor network has a soft update for 
every agent's critic network update (so n soft updates every learning cycle, where n is the total number of agents).

We also include Double DQN as part of our learning process, which stabilizes our learning by calculating our estimated best action from the next state using the local q network, 
and using those actions to calculate estimated next state reward from our target q network see [ref: Double DQN Paper](https://arxiv.org/abs/1509.06461).

The chosen neural network configuration for both our actor networks is a MLP with two hidden layers (default number of nodes 512 -> 256).
The chosen neural network configuration for all critic networks is a MLP with three hidden layers (default number of nodes 512 -> 256 -> 128).
All networks use batch normalization on the input parameters.

Hyperparameters used for this approach are provided in the hyperparameters.py file

## Current results

Through applying the above learning agent, we are able to achieve for a single agent a target score (averaged over the prior 100 episodes) of 0.5 after 741 episodes.  The results of our scores through successive training episodes are as shown:

![Epoch Scores](/common/images/score_by_episode.png "Epoch Scores")

## Areas for improvement

- More exhaustive hyperparameter tuning
- Testing against a decoupled actor network (one for each agent)
- Exploration of MADDPG techniques such as [MAPDDPG-GCPN](https://arxiv.org/pdf/1810.09206.pdf)
(extra actor network generates samples to store in replay buffer for critic usage)
