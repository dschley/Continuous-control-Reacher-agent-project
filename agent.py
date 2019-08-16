import random
from collections import namedtuple, deque
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim as optim

seed = 42069
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d4 = None

####################
# Hyper parameters:#
####################

# OUNoise hyperparams
mu = 0
theta = 0.15
sigma = 0.2

# Memory hyperparams
buffer_size = 100000
batch_size = 16

# Update hyperparams
gamma = 0.99
tau = 0.01
alr = 1e-4
clr = 1e-3

class Agent():
    '''
    Implementation where there is one agent per... agent
    
    TODO: maybe look into having just one "MultiAgent"
    and just reading the whole state space as one big tensor to act() 
    and do like an "addAll()" from the resulting env step in step()
    '''
    def __init__(self, state_size, action_size):
        global d4
        if d4 == None:
            d4 = D4PGBrain(state_size, action_size)
            
        self.d4 = d4
            
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        action = self.d4.actor_local(state)
        action = action.detach().numpy()
        action = action + self.d4.noise_state
        
        return action
        
        
    def step(self, state, action, reward, next_state, done):
        self.d4.memory.add(state, action, reward, next_state, done)
    
class MultiAgent():
    def __init__(self, state_size, action_size):
        self.d4 = D4PGBrain(state_size, action_size)
        
    def act(self, states):
        states = torch.FloatTensor(states)
        
        actions = self.d4.actor_local(states)
        actions = actions.detach().numpy()
        actions = actions + self.d4.noise_state
        
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        self.d4.memory.addAll(states, actions, rewards, next_states, dones)

class D4PGBrain():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.critic_local = Critic(self.state_size, self.action_size).to(device)
        self.critic_target = Critic(self.state_size, self.action_size).to(device)
        
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        
        self.actor_local = Actor(self.state_size, self.action_size).to(device)
        self.actor_target = Actor(self.state_size, self.action_size).to(device)
        
        self.noise = OUNoise(self.action_size, mu, theta, sigma)
        self.noise_state = self.noise.sample()
        
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
        self.critic_loss_func = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=alr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=clr)

        print('D4Brain created!')
        
    def save_state(self):
        torch.save(self.actor_local.state_dict(), 'actor_local.pth')
        torch.save(self.critic_local.state_dict(), 'critic_local.pth')
        
        
    def next_timestep(self):
        self.noise_state = self.noise.sample()
        
    def new_episode(self):
        self.noise.reset()
        
    def learn(self):
        experiences = self.memory.sample(batch_size)
        states, actions, rewards, next_states, _ = map(list, zip(*experiences))
#         print('actions: ',actions)
#         print('states: ',states)
#         print('actions[0]: ',actions[0])
#         print('states[0]: ',states[0])
#         print('actions[0][0]: ',actions[0][0])
#         print('actions[0].shape: ',actions[0].shape)
#         print('actions[0,0].shape: ',actions[0][0].shape)
#         print('states[0].shape: ',states[0].shape)

        
        states = torch.FloatTensor(states)

        actions = [action[0] for action in actions]
        actions = torch.FloatTensor(actions)
                
        rewards = torch.FloatTensor(rewards).view(states.shape[0],1)
        
        next_states = torch.FloatTensor(next_states)
        
#         print(states, states.shape)
#         print(actions, actions.shape)
        
        # Critic loss
        Qvals = self.critic_local(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach())
#         print('nq',next_Q.shape)
#         print('rwd',rewards.shape)
#         print('rwd2',rewards.view(16,1).shape)
        Qprime = rewards + gamma * next_Q
        critic_loss = self.critic_loss_func(Qvals, Qprime.detach())
        
        # Actor loss
        actor_loss = -self.critic_local(states, self.actor_local(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)
        
    def soft_update(self, local_model, target_model):
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1 - tau))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
        '''
        last project's implementation:
        state input:
        fc:32:relu
        fc:64:relu
        
        action input:
        fc:32:relu
        fc:64:relu
        
        merge individual networks and follow up with:
        relu
        fc:1:natural
        '''

    def forward(self, state, action):
        '''
        state_action = np.concatenate((state, action))
        state_action = torch.from_numpy(state_action).float().unsqueeze(0).to(device)
        '''
        state_action = torch.cat([state, action], 1)
        
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Actor, self).__init__()
#         self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        '''
        last project's implementation:
        fc:32:relu
        fc:64:relu
        fc:32:relu
        fc:out:sigmoid (which was multiplied by action range and added to action min)
        '''
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.tanh(self.fc4(x))
    

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def addAll(self, states, actions, rewards, next_states, dones):
        exps = [self.experience(state, action, reward, next_state, done) for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones)]
        for exp in exps:
            self.memory.append(exp)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)