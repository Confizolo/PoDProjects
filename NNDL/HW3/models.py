from collections import deque
from turtle import pen 
from tqdm import tqdm
from torch import nn
import torch
import random
import numpy as np

import pytorch_lightning as pl

class ReplayMemory(object):
    """
    Replay memory class

    Functions:
    -----------

    __init__ : Instantiate the ReplayMemory object

    push: Push new state, action, next_state and reward inside the memory of the object

    sample: Get a sample of predefined batch size from the memory

    __len__: Return the size of the memory

    """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity) # define a queue with maxlen "capacity"

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) # get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        return random.sample(self.memory, batch_size) # randomly select "batch_size" samples

    def __len__(self):
        return len(self.memory) # return the number of samples currently stored in the memory

class DQN(nn.Module):
    """
    DQN object used to predict q values in the state action space.

    """
    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        # Fully connected network
        self.linear = nn.Sequential(
            nn.Linear(state_space_dim, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,action_space_dim)
                )

    def forward(self, x):
        return self.linear(x)


def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
        
    # sample data from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)

    # create tensors for each element of the batch
    states      = torch.tensor([s[0] for s in batch], dtype=torch.float32)
    actions     = torch.tensor([s[1] for s in batch], dtype=torch.int64)
    rewards     = torch.tensor([s[3] for s in batch], dtype=torch.float32)

    # compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32) # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

    # compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states)
    # select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1))

    # compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0]

    # compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1) # Set the required tensor shape

    # compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # optimize the model
    optimizer.zero_grad()
    loss.backward()

    # apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()

def train_loop_pole(env,policy_net,target_net,exploration_profile, policy,replay_mem,hyper, render = False, early_stopping_pars = [None,None], pen = lambda x: 0):

    # architecture variables of the network
    bad_state_penalty = hyper["bad_state_penalty"]
    min_samples_for_training = hyper["min_samples_for_training"]
    gamma = hyper["gamma"]
    optimizer = hyper["optimizer"]
    lr = hyper["lr"]
    loss_fn = hyper["loss_fn"]
    batch_size = hyper["batch_size"]
    target_net_update_steps = hyper["target_net_update_steps"]

    scores = np.zeros(len(exploration_profile))
    for episode_num, tau in enumerate(tqdm(exploration_profile)):

        # reset the environment and get the initial state
        state = env.reset()

        # reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        done = False
        
        # go on until the pole falls off
        while not done:

            # choose the action following the policy
            action, q_values = policy(policy_net, state, tau)
            
            # apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
            next_state, reward, done, info = env.step(action)

            reward = reward + pen(state,env)
            # update the final score (+1 for each step)
            score += 1

            # apply penalty for bad state
            if done: # if the pole has fallen down 
                reward += bad_state_penalty
                next_state = None

            # update the replay memory
            replay_mem.push(state, action, next_state, reward)

            # update the network
            if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
                update_step(policy_net, target_net, replay_mem, gamma, optimizer(policy_net.parameters(), lr=lr[episode_num]), loss_fn, batch_size)

            if render == True:
                # visually render the environment 
                env.render()

            # set the current state for the next iteration
            state = next_state

        # increasing network score
        scores[episode_num]+=score

        # update the target network every target_net_update_steps episodes
        if episode_num % target_net_update_steps == 0:
            target_net.load_state_dict(policy_net.state_dict()) # this will copy the weights of the policy network to the target network

        if (early_stopping_pars[0] != None) &  (early_stopping_pars[1] != None):  
            if np.asarray(scores[max(0,-early_stopping_pars[0]+episode_num):episode_num]).mean() >= early_stopping_pars[1]:
                break

    # print the final score
    print(f"FINAL SCORE: {score} - Temperature: {tau}") # Print the final score

    env.close()

    return policy_net, target_net, scores

def train_loop_lander(env,policy_net,target_net,exploration_profile, policy,replay_mem,hyper, render = False, early_stopping_pars = [None,None], pen = lambda x: 0):

    # architecture variables of the network
    min_samples_for_training = hyper["min_samples_for_training"]
    gamma = hyper["gamma"]
    optimizer = hyper["optimizer"]
    lr = hyper["lr"]
    loss_fn = hyper["loss_fn"]
    batch_size = hyper["batch_size"]
    target_net_update_steps = hyper["target_net_update_steps"]

    scores = np.zeros(len(exploration_profile))
    for episode_num, tau in enumerate(tqdm(exploration_profile)):

        # reset the environment and get the initial state
        state = env.reset()
        # reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        done = False

        
        # Go on until the lander falls or time ends
        while not done:

            # choose the action following the policy
            action, q_values = policy(policy_net, state, tau)
            
            # apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
            next_state, reward, done, info = env.step(action)

            reward = reward + pen(state,env)
            # update the final score (+1 for each step)
            score += reward

            # update the replay memory
            replay_mem.push(state, action, next_state, reward)

            # update the network
            if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
                update_step(policy_net, target_net, replay_mem, gamma, optimizer(policy_net.parameters(), lr=lr[episode_num]), loss_fn, batch_size)

            if render == True:
                # visually render the environment (disable to speed up the training)
                env.render()

            # set the current state for the next iteration
            state = next_state

        # update the score vector
        scores[episode_num]+=score

        # update the target network every target_net_update_steps episodes
        if episode_num % target_net_update_steps == 0:
            target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

        if (early_stopping_pars[0] != None) &  (early_stopping_pars[1] != None):  
            if np.asarray(scores[max(0,-early_stopping_pars[0]+episode_num):episode_num]).mean() >= early_stopping_pars[1]:
                break
        # print the final score
        # print(f"FINAL SCORE: {score} - Temperature: {tau}") 
    env.close()

    return policy_net, target_net, scores


def test_loop(env, policy, policy_net, num_ep= 10):

    for num_episode in range(num_ep): 
        # reset the environment and get the initial state
        state = env.reset()
        # reset the score
        score = 0
        done = False
        # go on until the episode ends
        while not done:

            # choose the best action (temperature/epsilon 0)
            action, q_values = policy(policy_net, state, 0)
            # apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
            next_state, reward, done, info = env.step(action)
            # visually render the environment
            env.render()
            # update the final score (+1 for each step)
            score += reward 
            # set the current state for the next iteration
            state = next_state
            # check if the episode ended (the pole fell down)
            # print the final score
            
        print(f"Test {num_episode} final score:  {score}") 
    env.close()