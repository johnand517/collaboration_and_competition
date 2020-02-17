import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

import hyperparameters as hp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, seed):
        """Initialize an Agent object.

        :param state_size: (int) dimension of state space
        :param action_size: (int) dimension of action space
        :param seed: (int) random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hp.actor_learning_rate)

        # Critic Network (w/ Target Network) for each agent
        self.critic_local = []
        self.critic_target = []
        self.critic_optimizer = []
        for i in range(num_agents):
            self.critic_local.append(Critic(state_size, action_size, num_agents, seed).to(device))
            self.critic_target.append(Critic(state_size, action_size, num_agents, seed).to(device))
            self.critic_optimizer.append(optim.Adam(self.critic_local[i].parameters(), lr=hp.critic_learning_rate,
                                           weight_decay=hp.weight_decay))

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, hp.buff_size, hp.batch_size, seed)
    
    def step(self, state, action, reward, next_state, done):
        """Increments

        :param state: (array_like) the current environment state
        :param action: (array_like) the determined action by our agent
        :param reward: the reward given the current state and chosen action
        :param next_state: the determined next state given our current state and action
        :param done: (bool) determines if we have reached a terminal state

        """
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > hp.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, hp.Gamma)

    def act(self, state, add_noise=True):
        """Returns actions given the provided states for the
        given policy

        :param state: (array_like) current state
        :param add_noise: (bool) add OU noise to our action selection

        :return: an action value
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, exps, gamma):
        """ Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param exps: (array_like) list of experience tuples
        :param gamma: (float) hyperparameter discount factor
        """
        states, actions, rewards, next_states, dones = exps

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        # Reshape all of our parameters
        actions = actions.reshape(self.memory.batch_size, self.num_agents*self.action_size)
        actions_next = actions_next.reshape(self.memory.batch_size, self.num_agents*self.action_size)
        next_states =  next_states.reshape(self.memory.batch_size, self.num_agents, self.state_size)
        states_reshape = states.reshape(self.memory.batch_size, self.num_agents, self.state_size)
        rewards = rewards.reshape(self.memory.batch_size, self.num_agents, 1)
        dones = dones.reshape(self.memory.batch_size, self.num_agents, 1)

        critic_loss = []
        for i in range(self.num_agents):
            # ---------------------------- update critic ---------------------------- #
            Q_targets_next = self.critic_target[i](next_states[:, i], actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards[:, i] + (gamma * Q_targets_next * (1 - dones[:, i]))
            # Compute critic loss
            Q_expected = self.critic_local[i](states_reshape[:, i], actions)
            critic_loss.append(F.mse_loss(Q_expected, Q_targets))
            # Minimize the loss
            self.critic_optimizer[i].zero_grad()
            critic_loss[i].backward(retain_graph=True)
            clip_grad_norm_(self.critic_local[i].parameters(), 1)
            self.critic_optimizer[i].step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actions_pred = actions_pred.reshape(self.memory.batch_size, self.num_agents*self.action_size)
            actor_loss = -self.critic_local[i](states_reshape[:, i], actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # ----------------------- update target critic networks ----------------------- #
            self.soft_update(self.critic_local[i], self.critic_target[i], hp.tau)

            # ------------------ update target actor network ----------------------- #
            self.soft_update(self.actor_local, self.actor_target, hp.tau)

    def soft_update(self, local_model, target_model, tau):
        """Updates DQN model parameters

        :param local_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :param tau: (float) interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=hp.noise_mu, theta=hp.noise_theta, sigma=hp.noise_sigma):
        """Initialize parameters and noise process.

        :param size: (int)
        :param seed:  (int)
        :param mu: (float)
        :param theta: (float)
        :param sigma: (float)
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object

        :param action_size: (int) dimension of action space
        :param buffer_size: (int) the size of our ReplayBuffer
        :param batch_size: (int) training size to pull from buffer
        :param seed: (int) random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory

        :param state: (array_like) the current environment state
        :param action: (array_like) the determined action by our agent
        :param reward:  the reward given the current state and chosen action
        :param next_state:  the determined next state given our current state and action
        :param done: (bool) determines if we have reached a terminal state
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly selects a sample of experiences from memory

        :return: (tuple) a previously learned experience
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory.

        :return: (int) size of memory buffer
        """
        return len(self.memory)