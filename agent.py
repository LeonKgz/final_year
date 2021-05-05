from LoRaParameters import LoRaParameters

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import torch
import torch.nn.utils.weight_norm as weight_norm

# Deep Reinforcement Learning Agent section
class LearningAgent:

    def __init__(self, env, alpha=0.001, gamma=0.1, epsilon=0.5, sarsa=False):

        self.type = "Q Learning"
        self.sarsa = sarsa
        self.nodes = []

        # Create a Q-network, which predicts the q-value for a particular state.
        self.env = env

        tps = len(LoRaParameters.TRANSMISSION_POWERS)
        sfs = len(LoRaParameters.SPREADING_FACTORS)
        chs = len(LoRaParameters.DEFAULT_CHANNELS)


        self.index_to_action = []
        self.action_to_index = {}

        curr_idx = 0
        for tp in LoRaParameters.TRANSMISSION_POWERS:
            for sf in LoRaParameters.SPREADING_FACTORS:
                for ch in LoRaParameters.DEFAULT_CHANNELS:
                    tensor = torch.tensor([tp, sf, ch])
                    self.index_to_action.append(tensor)
                    self.action_to_index[tensor] = curr_idx
                    curr_idx += 1

        self.action_size = 90
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.losses = {}

        self.q_table = torch.Tensor(tps, sfs, chs, self.action_size)

    def convert_state_to_index(self, s):
        tp_S, sf_S, channel_S = s
        return (LoRaParameters.TRANSMISSION_POWERS.index(tp_S),
                LoRaParameters.SPREADING_FACTORS.index(sf_S),
                LoRaParameters.DEFAULT_CHANNELS.index(channel_S))

    def assign_nodes(self, nodes):

        if (len(nodes) == 0):
            raise Exception("Empty cluster of nodes is being passed to the Learning Agent")

        self.nodes = nodes
        for node in nodes:
            node.assign_learning_agent(agent=self)

    def choose_next_action(self, curr_s):
        # curr_s = self.current_s()

        output = self.q_table[self.convert_state_to_index(curr_s)]
        # max_action refers to the maximum reward value among the possible actions that can be taken
        max_action = torch.max(output)

        pi = []
        max_found = False

        for a in output:
            if (a == max_action and not max_found):
                pi.append(1 - self.epsilon + (self.epsilon / self.action_size))
                max_found = True
            else:
                pi.append(self.epsilon / self.action_size)

        pi[-1] = 1 - sum(pi[:-1])

        if (pi[-1] < 0):
            print(f"Last value is negative: {pi[-1]}")
            pi[-1] = 0
            # TODO catch Value errors for weird pi values and report them

        try:
            selected_value = np.random.choice(output.detach().numpy(), p=pi)
        except ValueError:
            print("Problem with the pi values!")
            exit(-1)

        if math.isnan(selected_value):
            print("NAN ENCOUNTERED")
            return random.choice(self.index_to_action)

        ret = self.index_to_action[output.detach().numpy().tolist().index(selected_value)]
        return ret

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    # transition = (s, a, r, s')
    def train_q_network(self, transition):
        s, a, reward, next_s = transition

        if self.sarsa:
            next_a = self.choose_next_action(next_s)
            next_a = self.action_to_index[next_a]

        s = self.convert_state_to_index(s)
        a = self.action_to_index[a]
        next_s = self.convert_state_to_index(next_s)

        q_s_a = self.q_table[s][a]

        if math.isnan(q_s_a):
            q_s_a = 0

        max_q_s_a = torch.max(self.q_table[next_s][a])

        if (self.sarsa):
            next_q_s_a = self.q_table[next_s][next_a]
            self.q_table[s][a] = q_s_a + self.alpha * (reward + self.gamma * next_q_s_a - q_s_a)
        else:
            self.q_table[s][a] = q_s_a + self.alpha * (reward + self.gamma * max_q_s_a - q_s_a)



# Deep Reinforcement Learning Agent section
class DeepLearningAgent:

    def __init__(self, env, depth, state_space_dimensions, lr=0.001, gamma=0.1, epsilon=0.5, sarsa=False, replay_buffer=False):

        self.type = "Deep Q Learning"
        self.nodes = []
        self.sarsa = sarsa

        # Create a Q-network, which predicts the q-value for a particular state.
        self.env = env
        self.q_network = Network(input_dimensions=state_space_dimensions, output_dimensions=90, depth=depth)

        self.index_to_action = []
        self.action_to_index = {}
        curr_idx = 0
        for tp in LoRaParameters.TRANSMISSION_POWERS:
            for sf in LoRaParameters.SPREADING_FACTORS:
                for ch in LoRaParameters.DEFAULT_CHANNELS:
                    tensor = torch.tensor([tp, sf, ch])
                    self.index_to_action.append(tensor)
                    self.action_to_index[tensor] = curr_idx
                    curr_idx += 1

        self.action_size = 90
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

        self.losses = {}

        self.replay_buffer = replay_buffer
        self.buffer = ReplayBuffer(50)
        self.batch_size = 10
        self.lr = lr

    def assign_nodes(self, nodes):

        if (len(nodes) == 0):
            raise Exception("Empty cluster of nodes is being passed to the Learning Agent")

        self.nodes = nodes
        for node in nodes:
            node.assign_learning_agent(agent=self)

    def choose_next_action(self, curr_s):
        # curr_s = self.current_s()
        output = self.q_network(curr_s)
        max_action = torch.max(output)
        pi = []
        max_found = False
        for a in output:
            if (a == max_action and not max_found):
                pi.append(1 - self.epsilon + (self.epsilon / self.action_size))
                max_found = True
            else:
                pi.append(self.epsilon / self.action_size)

        pi[-1] = 1 - sum(pi[:-1])

        if (pi[-1] < 0):
            print(f"Last value is negative: {pi[-1]}")
            pi[-1] = 0
            # TODO catch Value errors for weird pi values and report them

        try:
            selected_value = np.random.choice(output.detach().numpy(), p=pi)
        except ValueError:
            print("Problem with the pi values!")
            exit(-1)

        ret = self.index_to_action[output.detach().numpy().tolist().index(selected_value)]
        return ret

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):

        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()

        if not self.replay_buffer:

            # Calculate the loss for this transition.
            loss = self._calculate_loss(transition)

        else:
            # Implement replay buffer
            self.buffer.push(transition)

            if self.buffer.sizeof() < self.batch_size:
                return

            transitions = self.buffer.sample(self.batch_size)
            loss = self._calculate_batch_loss(transitions)

        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()

        # Take one gradient step to update the Q-network.
        self.optimiser.step()

        # Return the loss as a scalar
        return loss.item()

    # def _calculate_batch_loss(self, transitions):
    #     transposed = list(zip(*transitions))
    #
    #     states = [s for s in transposed[0]]
    #     actions = [a for a in transposed[1]]
    #     rewards = [r for r in transposed[2]]
    #
    #     next_states = [ns for ns in transposed[3]]
    #
    #     targets = []
    #     predicts = []
    #
    #     loss = torch.nn.MSELoss()
    #     loss_acc = 0
    #
    #     for i in range(self.batch_size):
    #         targets.append(rewards[i] + self.gamma * torch.max(self.q_network(next_states[i])))
    #         predicted = self.q_network(states[i])
    #         predicts.append(predicted[self.action_to_index[actions[i]]])
    #         loss_acc += loss(targets[i], predicts[i])
    #
    #     loss = loss_acc / self.batch_size
    #     self.losses[self.env.now] = loss
    #
    #     # expected_batch = self.q_network(next_state_batch).max(1)[0]
    #     # target = reward_batch + self.gamma * expected_batch
    #     # state_action_values = self.q_network(state_batch).gather(1, action_batch)
    #
    #     return loss

    def _calculate_batch_loss(self, transitions):
        # transition — (s, a, r, s')

        transposed = list(zip(*transitions))

        states = torch.cat([s.unsqueeze(0) for s in transposed[0]])
        actions = torch.cat([torch.tensor([self.action_to_index[a]], dtype=torch.int64).unsqueeze(0) for a in transposed[1]])
        rewards = torch.cat([torch.tensor([r]).unsqueeze(0) for r in transposed[2]])

        next_states = torch.cat([ns.unsqueeze(0) for ns in transposed[3]])

        if self.sarsa:
            ### TODO This line is too complicated, need to implement choose_next_action for batch input ###
            next_actions = torch.cat([torch.tensor([self.action_to_index[self.choose_next_action(ns)]], dtype=torch.int64).unsqueeze(0) for ns in transposed[3]])
            target = rewards + self.gamma * self.q_network(next_states).gather(1, next_actions)
        else:
            ### the same operation broken down ###
            # temp = self.q_network(next_states)
            # temp = torch.max(temp, dim=1)[0]
            # temp = temp.detach()
            # temp = temp.squeeze(0).view(self.batch_size, 1)
            temp = torch.max(self.q_network(next_states), dim=1)[0].detach().squeeze(0).view(self.batch_size, 1)
            target = rewards + temp * self.gamma

        predicted = self.q_network(states)
        predicted = predicted.gather(1, actions)

        loss = torch.nn.MSELoss()
        loss = loss(target, predicted)
        # loss_acc = 0
        #
        # for i in range(self.batch_size):
        #     targets.append(rewards[i] + self.gamma * torch.max(self.q_network(next_states[i])))
        #     predicted = self.q_network(states[i])
        #     predicts.append(predicted[self.action_to_index[actions[i]]])
        #     loss_acc += loss(targets[i], predicts[i])
        #
        # loss = loss_acc / self.batch_size
        self.losses[self.env.now] = loss

        # expected_batch = self.q_network(next_state_batch).max(1)[0]
        # target = reward_batch + self.gamma * expected_batch
        # state_action_values = self.q_network(state_batch).gather(1, action_batch)

        return loss


    # Function to calculate the loss for a particular transition.
    # transition = (s, a, r, s')
    def _calculate_loss(self, transition):
        s, a, reward, next_s = transition
        if self.sarsa:
            next_a = self.choose_next_action(next_s)
            target = reward + self.gamma * self.q_network(next_s)[self.action_to_index[next_a]]
        else:
            target = reward + self.gamma * torch.max(self.q_network(next_s))
        # is_it = a in self.index_to_action
        # print(is_it)
        predicted = self.q_network(s)
        predicted = predicted[self.action_to_index[a]]
        loss = torch.nn.MSELoss()
        loss = loss(target, predicted)
        self.losses[self.env.now] = loss
        return loss

class Network(torch.nn.Module):

    def __init__(self, input_dimensions, output_dimensions, depth):
        super(Network, self).__init__()

        layers = []

        layers.append(torch.nn.Linear(in_features=input_dimensions, out_features=100))
        layers.append(torch.nn.ReLU())
        for i in range(depth-1):
            layers.append(weight_norm(torch.nn.Linear(in_features=100, out_features=100), name='weight'))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(in_features=100, out_features=output_dimensions))

        self.network = torch.nn.Sequential(*layers)

        # self.network = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=input_dimensions, out_features=100),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=100, out_features=100),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=100, out_features=output_dimensions)
        # )

    def forward(self, input):
        return self.network(input)

import random

class ReplayBuffer():

    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.curr_pointer = 0

    def push(self, transition):

        # TODO change implementation
        if len(self.buffer) < self.size:
            self.buffer.append(None)

        self.buffer[self.curr_pointer] = transition
        self.curr_pointer = (self.curr_pointer + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sizeof(self):
        return len(self.buffer)
