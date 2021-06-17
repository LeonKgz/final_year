### START ###

from LoRaParameters import LoRaParameters

import numpy as np
import random
import math
import torch
import torch.nn.utils.weight_norm as weight_norm

class LearningAgent:

    def __init__(self, env, config):
        self.nodes = []
        self.next_actions = {}
        self.config = config
        self.env = env
        self.location = None

        self.sarsa = config["sarsa"]
        self.double_deep = config["double_deep"]
        self.replay_buffer = config["replay_buffer"]
        self.mc = config["mc"]
        self.device = config["device"]
        self.gamma = config["gamma"]
        self.depth = config["depth"]

        # epsilon and alpha will be given (starting!) values later when assigning nodes
        self.epsilon = 0.5
        self.alpha = 0.5

        self.index_to_action = []
        self.action_to_index = {}
        self.losses = {}

        curr_idx = 0

        if config["slow_tp"]:
            self.tp_range = range(-1, 2)
        else:
            self.tp_range = LoRaParameters.TRANSMISSION_POWERS

        if config["slow_sf"]:
            self.sf_range = range(-1, 2)
        else:
            self.sf_range = LoRaParameters.SPREADING_FACTORS

        if config["slow_channel"]:
            self.channel_range = range(-1, 2)
        else:
            self.channel_range = LoRaParameters.DEFAULT_CHANNELS

        for tp in self.tp_range:
            for sf in self.sf_range:
                for ch in self.channel_range:
                    tensor = torch.tensor([tp, sf, ch])
                    self.index_to_action.append(tensor)
                    self.action_to_index[tensor] = curr_idx
                    curr_idx += 1
        self.action_size = curr_idx

        # start with epsilon = 1 / 2
        # this counter determines the current epsilon value
        self.epsilon_value_counter = 2
        # these counters determine when the epsilon value is updated
        self.epsilon_update_counter = 0
        self.epsilon_update_rate = None

        # start with learning rate 1 / 2^1
        # this counter determines the current alpha value
        self.alpha_value_counter = 1
        # these counters determine when the epsilon value is updated
        self.alpha_update_counter = 0
        self.alpha_update_rate = None

        # Naive tabular implementation
        if (not self.config["deep"]):
            tps = len(LoRaParameters.TRANSMISSION_POWERS)
            sfs = len(LoRaParameters.SPREADING_FACTORS)
            chs = len(LoRaParameters.DEFAULT_CHANNELS)
            self.q_table = torch.Tensor(tps, sfs, chs, self.action_size)

        if (self.config["deep"]):
            # Initialize Q-network and Target network.
            # self.q_network = Network(input_dimensions=len(config["state_space"]), output_dimensions=self.action_size, depth=depth)
            self.q_network = Network(input_dimensions=len(config["state_space"]), output_dimensions=self.action_size,
                                     depth=self.depth, width=self.config["width"]).to(config["device"])
            # self.target_network = Network(input_dimensions=len(config["state_space"]), output_dimensions=self.action_size, depth=depth)
            self.target_network = Network(input_dimensions=len(config["state_space"]),
                                          output_dimensions=self.action_size, depth=self.depth, width=self.config["width"]).to(config["device"])
            self.target_network.load_state_dict(self.q_network.state_dict())

            self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.alpha)

            self.batch_size = -1
            self.buffer = ReplayBuffer()
            self.target_update_counter = 0
            self.target_update_rate = -1
            self.replay_buffer_size =-1

    def assign_nodes(self, nodes):
        num_nodes = len(nodes)
        if (num_nodes == 0):
            raise Exception("Empty cluster of nodes is being passed to the Learning Agent")

        self.nodes = nodes
        for node in nodes:
            node.assign_learning_agent(agent=self)
            if (self.sarsa):
                state = node.current_s()
                if (self.config["deep"]):
                    state = state.to(self.config["device"])
                self.next_actions[node.id] = self.choose_next_action(state, node_id=None)

        self.epsilon_update_rate = num_nodes * self.config["epsilon_decay_rate"]
        self.alpha_update_rate = num_nodes * self.config["alpha_decay_rate"]

        self.epsilon_update_rate = int(num_nodes * self.config["epsilon_decay_rate"])
        self.alpha_update_rate = int(num_nodes * self.config["alpha_decay_rate"])

        self.batch_size = int(num_nodes * self.config["replay_buffer_scale"])
        self.target_update_rate = int(num_nodes * self.config["target_update_rate"])

        if (num_nodes == 1):
            self.epsilon_update_rate = 1
            self.alpha_update_rate = 1
            self.target_update_rate = 1
            self.batch_size = 1

        if (self.config["deep"]):
            self.replay_buffer_size = 3 * self.batch_size
            self.buffer.set_size(self.replay_buffer_size)

    def convert_state_to_index(self, state):
        tp_S, sf_S, channel_S = state
        return (LoRaParameters.TRANSMISSION_POWERS.index(tp_S),
                LoRaParameters.SPREADING_FACTORS.index(sf_S),
                LoRaParameters.DEFAULT_CHANNELS.index(channel_S))

    def choose_next_action(self, curr_state, node_id, need_new_value=False):

        # this branch is executed at the start, when nodes are assigned to this lreaning agent
        # a random action will suffice
        if (self.sarsa and node_id == None):
            return self.index_to_action[(int) (np.random.rand() * len(self.index_to_action))]

        # this branch is executed when node is requesting next action (in SARSA setting)
        # next action is already stored in the next_actions array, which is updated within _calculate_loss
        if (self.sarsa and not need_new_value):
            return self.next_actions[node_id]

        if (self.config["deep"]):
            curr_state = curr_state.to(self.device)
            output = self.q_network(curr_state)
            output = output.cpu()
        else:
            output = self.q_table[self.convert_state_to_index(curr_state)]

        max_action = torch.max(output)
        pi = []
        max_found = False
        for a in output:
            if (a == max_action and not max_found):
                pi.append(1 - self.epsilon + (self.epsilon / self.action_size))
                max_found = True
            else:
                pi.append(self.epsilon / self.action_size)

        if (self.config["GLIE"] ):
            if (self.config["epsilon_decay_rate"] != -1):
                self.epsilon_update_counter += 1
                if self.epsilon_update_counter == self.epsilon_update_rate:
                    # Update epsilon to ensure convergence according to GLIE
                    self.epsilon = 1 / self.epsilon_value_counter
                    self.epsilon_value_counter += 1
                    self.epsilon_update_counter = 0
            else:
                self.epsilon_value_counter += 1
                self.epsilon = 1 / self.epsilon_value_counter

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

        if (not self.config["deep"] and math.isnan(selected_value)):
            # print("NAN ENCOUNTERED")
            return random.choice(self.index_to_action)

        ret = self.index_to_action[output.detach().numpy().tolist().index(selected_value)]
        return ret

    def choose_next_action_average_of_two(self, curr_state):

        if (not self.config["sarsa"] or not self.config["double_deep"] or not self.config["deep"]):
            raise Exception("choose_next_action_average_of_two() can only be called in double deep SARSA")

        # TODO implement for tabular
        # if (self.config["deep"]):
        #   curr_state = curr_state.to(self.device)
        #   output = self.q_network(curr_state)
        #   output = output.cpu()
        # else:
        #     output = self.q_table[self.convert_state_to_index(curr_state)]

        curr_state = curr_state.to(self.device)

        output_a = self.q_network(curr_state)
        output_a = output_a.cpu()

        output_b = self.target_network(curr_state)
        output_b = output_b.cpu()

        # max_action = torch.max(output_a) + torch.max(output_b)
        max_action = torch.max(output_a + output_b).detach()

        pi = []
        max_found = False
        # self.index_to_action[output_a.detach().numpy().tolist().index(selected_value)]
        for a in output_a:
            a_index = output_a.detach().numpy().tolist().index(a)
            if (a.detach() + output_b[a_index].detach() == max_action and not max_found):
                pi.append(1 - self.epsilon + (self.epsilon / self.action_size))
                max_found = True
            else:
                pi.append(self.epsilon / self.action_size)

        if (self.config["GLIE"] ):
            if (self.config["epsilon_decay_rate"] != -1):
                self.epsilon_update_counter += 1
                if self.epsilon_update_counter == self.epsilon_update_rate:
                    # Update epsilon to ensure convergence according to GLIE
                    self.epsilon = 1 / self.epsilon_value_counter
                    self.epsilon_value_counter += 1
                    self.epsilon_update_counter = 0
            else:
                self.epsilon_value_counter += 1
                self.epsilon = 1 / self.epsilon_value_counter

        pi[-1] = 1 - sum(pi[:-1])

        if (pi[-1] < 0):
            print(f"Last value is negative: {pi[-1]}")
            pi[-1] = 0
            # TODO catch Value errors for weird pi values and report them

        try:
            selected_value = np.random.choice(output_a.detach().numpy(), p=pi)
        except ValueError:
            print("Problem with the pi values!")
            exit(-1)

        if (not self.config["deep"] and math.isnan(selected_value)):
            # print("NAN ENCOUNTERED")
            return random.choice(self.index_to_action)

        ret = self.index_to_action[output_a.detach().numpy().tolist().index(selected_value)]
        return ret

    # Expected return value used in the Expected SARSA algorithm
    def expected_action_value(self, state):
        if (not self.config["expected_sarsa"]):
            raise Exception("expected_action_value() should only be called in the Expected SARSA type algorithm")
        ret = 0
        if (self.config["deep"]):
            state = state.to(self.device)
            if (self.config["double_deep"]):
                output = self.target_network(state)
            else:
                output = self.q_network(state)
            output = output.cpu()
        else:
            output = self.q_table[self.convert_state_to_index(state)]

        max_action = torch.max(output)
        max_found = False
        for a in output:
            if (a == max_action and not max_found):
                pi = 1 - self.epsilon + (self.epsilon / self.action_size)
                max_found = True
                ret += pi * a
            else:
                pi = self.epsilon / self.action_size
                ret += pi * a

        return ret

    def train_q(self, transition, node_id):
        if (self.config["deep"]):
            self.deep_train_q_network(transition, node_id)
        else:
            self.tabular_train_q_network(transition, node_id)
        self.update_alpha()

    def update_alpha(self):
        if (self.config["Robbins-Monroe"]):
            if (self.config["alpha_decay_rate"] != -1):
                self.alpha_update_counter += 1
                if (self.alpha_update_counter == self.alpha_update_rate):
                    self.alpha = 1 / (2 ** self.alpha_value_counter)
                    if (self.config["deep"]):
                      self.optimiser.param_groups[0]['lr'] = self.alpha
                    self.alpha_value_counter += 1
                    self.alpha_update_counter = 0
            else:
                self.alpha = 1 / (2 ** self.alpha_value_counter)
                if (self.config["deep"]):
                  self.optimiser.param_groups[0]['lr'] = self.alpha
                self.alpha_value_counter += 1

    def deep_train_q_network(self, transition, node_id):
        self.optimiser.zero_grad()

        if not self.replay_buffer or self.sarsa:

            loss = self._calculate_loss(transition, node_id)

            if (self.sarsa and self.config["double_deep"]):

                # swap two tables
                if (np.random.uniform() > 0.5):
                    temp_ref = self.q_network
                    self.q_network = self.target_network
                    self.target_network = temp_ref
                    # self.q_network.load_state_dict(self.target_network.state_dict())
                    # self.target_network.load_state_dict(state_dict_temp)

        else:
            # Implement replay buffer (reminder: cannot implement replay buffer for SARSA!)
            self.buffer.push(transition)

            if self.buffer.sizeof() < self.batch_size:
                return

            transitions, positions = self.buffer.sample(self.batch_size)
            loss = self._calculate_batch_loss(transitions, positions)

            self.target_update_counter += 1

            if (self.double_deep):
                # Synchronize Q and Target Networks every now and then
                if self.target_update_counter == self.target_update_rate:
                    self.target_update_counter = 0
                    self.target_network.load_state_dict(self.q_network.state_dict())

        loss.backward()

        if len(self.optimiser.param_groups) > 1:
            raise Exception("Number of parameter groups for the optimizer is more than one")

        # self.update_alpha()
        self.optimiser.step()

        return loss.item()

    def tabular_train_q_network(self, transition, node_id):
        s, a, reward, next_s = transition

        if self.sarsa:
            # This is is a bug that performed as well as deep models
            # TODO: figure out WHY not taking actions performs so well
            # next_a = self.choose_next_action_epsilon_greedy(next_s, node_id=node_id)
            # next_a_index = self.action_to_index[next_a]

            next_a = self.choose_next_action(next_s, node_id=node_id, need_new_value=True)
            self.next_actions[node_id] = next_a
            next_a_index = self.action_to_index[next_a]

        s_index = self.convert_state_to_index(s)
        a_index = self.action_to_index[a]
        next_s_index = self.convert_state_to_index(next_s)

        q_s_a = self.q_table[s_index][a_index]

        if math.isnan(q_s_a):
            q_s_a = 0

        max_q_s_a = torch.max(self.q_table[next_s_index][a_index])

        if (self.sarsa):
            next_q_s_a = self.q_table[next_s_index][next_a_index]
            self.q_table[s_index][a_index] = q_s_a + self.alpha * (reward + self.gamma * next_q_s_a - q_s_a)
        if (self.mc):
            self.q_table[s_index][a_index] = q_s_a + self.alpha * (reward - q_s_a)
        else:
            self.q_table[s_index][a_index] = q_s_a + self.alpha * (reward + self.gamma * max_q_s_a - q_s_a)

    def _calculate_batch_loss(self, transitions, positions):
        # transition — (s, a, r, s')

        transposed = list(zip(*transitions))

        states = torch.cat([s.unsqueeze(0) for s in transposed[0]])
        actions = torch.cat(
            [torch.tensor([self.action_to_index[a]], dtype=torch.int64).unsqueeze(0) for a in transposed[1]])
        rewards = torch.cat([torch.tensor([r]).unsqueeze(0) for r in transposed[2]])

        next_states = torch.cat([ns.unsqueeze(0) for ns in transposed[3]])

        if self.sarsa:
            ### TODO This line is too complicated, need to implement choose_next_action for batch input ###
            raise Exception("Shouldn't reach this code! Cannot implement Replay Buffer for SARSA!")
            # next_actions = torch.cat(
            #     [torch.tensor([self.action_to_index[self.choose_next_action(ns)]], dtype=torch.int64).unsqueeze(0) for
            #      ns in transposed[3]])
            # target = rewards + self.gamma * self.q_network(next_states).gather(1, next_actions)
        else:
            ### the same operation broken down ###
            # temp = self.q_network(next_states)
            # temp = torch.max(temp, dim=1)[0]
            # temp = temp.detach()
            # temp = temp.squeeze(0).view(self.batch_size, 1)

            if (self.double_deep):

                # Simple implementation of Target Network optimization
                # temp = torch.max(self.target_network(next_states), dim=1)[0].detach().squeeze(0).view(self.batch_size, 1)
                temp = self.target_network(next_states).cpu()
                optimal_action_indexes = torch.argmax(temp, dim=1)

                temp = self.q_network(next_states).cpu()
                temp = torch.tensor([q[i] for (q, i) in zip(temp, optimal_action_indexes)]).squeeze(0).view(self.batch_size,
                                                                                                            1)

            else:

                temp = self.q_network(next_states).cpu()
                temp = torch.max(temp, dim=1)[0].detach().squeeze(0).view(self.batch_size, 1)

            target = rewards + temp * self.gamma

        predicted = self.q_network(states).cpu()
        predicted = predicted.gather(1, actions)

        # Commenting this code out, attempt to use squared loss value instead of absolute difference
        # relative to each other the values should be the same (abs() and pow(2) are similar operations)
        # delta = target - predicted
        # loss = delta.pow(2)

        # reduction="None" ensures loss is calculated across the whole batch
        # 10 losses are produced (in one tensor of size [10, 1])
        loss = torch.nn.MSELoss(reduction="none")
        loss = loss(target.float(), predicted.float())

        weights = loss + 1e-5
        weights = weights.squeeze(1).tolist()
        self.buffer.update_weights(positions, weights)

        loss = loss.mean()
        self.losses[self.env.now] = loss

        return loss

        # transition = (s, a, r, s')

    def _calculate_loss(self, transition, node_id):
        s, a, reward, next_s = transition
        if self.sarsa:
            if self.config["double_deep"]:
                if self.config["expected_sarsa"]:
                    next_a = self.choose_next_action_average_of_two(next_s)
                    self.next_actions[node_id] = next_a
                    target = reward + self.gamma * self.expected_action_value(next_s)
                else:
                    next_a = self.choose_next_action_average_of_two(next_s)
                    self.next_actions[node_id] = next_a
                    target = reward + self.gamma * self.target_network(next_s)[self.action_to_index[next_a]]
            if self.config["expected_sarsa"]:
                next_a = self.choose_next_action(next_s, node_id, need_new_value=True)
                self.next_actions[node_id] = next_a
                target = reward + self.gamma * self.expected_action_value(next_s)
            else:
                next_a = self.choose_next_action(next_s, node_id, need_new_value=True)
                self.next_actions[node_id] = next_a
                target = reward + self.gamma * self.q_network(next_s)[self.action_to_index[next_a]]
        else:
            if (self.double_deep):
                temp = self.target_network(next_s).cpu()
                target = reward + self.gamma * torch.max(temp)
            else:
                temp = self.q_network(next_s).cpu()
                target = reward + self.gamma * torch.max(temp)

        # is_it = a in self.index_to_action
        # print(is_it)
        predicted = self.q_network(s)
        # predicted = self.q_network(s).cpu()
        predicted = predicted[self.action_to_index[a]]
        loss = torch.nn.MSELoss()
        loss = loss(target.cpu(), predicted.cpu())
        self.losses[self.env.now] = loss
        return loss

class Network(torch.nn.Module):

    def __init__(self, input_dimensions, output_dimensions, depth, width):
        super(Network, self).__init__()

        layers = []

        layers.append(torch.nn.Linear(in_features=input_dimensions, out_features=width))
        layers.append(torch.nn.ReLU())
        for i in range(depth-1):
            layers.append(weight_norm(torch.nn.Linear(in_features=width, out_features=width), name='weight'))
            layers.append(torch.nn.ReLU(inplace=False))
        layers.append(torch.nn.Linear(in_features=width, out_features=output_dimensions))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = torch.nn.Sequential(*layers)
        # self.network = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=input_dimensions, out_features=100),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=100, out_features=100),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=100, out_features=output_dimensions)
        # )

    def forward(self, input):
        x = input.to(self.device)
        return self.network(input)

class ReplayBuffer():

    def __init__(self, alpha=0.5):
        self.size = 1
        self.buffer = []
        self.weights = []
        self.curr_pointer = 0
        self.curr_max_weight = 1 / self.size
        self.alpha = alpha

    def push(self, transition):

        # TODO change implementation
        if len(self.buffer) < self.size:
            self.buffer.append(None)
            self.weights.append(None)
        self.buffer[self.curr_pointer] = transition
        self.weights[self.curr_pointer] = self.curr_max_weight

        # self.buffer[self.curr_pointer] = (transition, self.curr_max_weight)
        self.curr_pointer = (self.curr_pointer + 1) % self.size

    def sample(self, batch_size):
        weights_prioritised = [w ** self.alpha for w in self.weights]
        weight_sum = sum(weights_prioritised)
        p_distribution = [(w / weight_sum) for w in weights_prioritised]
        p_distribution[-1] = 1 - sum(p_distribution[:-1])

        positions = np.random.choice(len(self.buffer), batch_size, p=p_distribution)
        # selected_value = np.random.choice(output.detach().numpy(), p=pi)
        batch = [self.buffer[i] for i in positions]

        return batch, positions

    def update_weights(self, positions, weights):
        # Here the weights might (most probably will) change so need to update the current max weight
        curr_max = self.curr_max_weight

        for p, w in zip(positions, weights):
            self.weights[p] = w
            if w > curr_max:
                curr_max = w

        self.curr_max_weight = curr_max

    def sizeof(self):
        return len(self.buffer)

    def set_size(self, replay_buffer_size):
       self.size = replay_buffer_size

# Old versions of methods

# class LearningAgent:
#
#     def __init__(self, env, config, alpha=0.001):
#
#         self.type = "Q Learning"
#         self.sarsa = config["sarsa"]
#         self.mc = config["mc"]
#         self.nodes = []
#         self.config = config
#         self.env = env
#
#         self.index_to_action = []
#         self.action_to_index = {}
#
#         curr_idx = 0
#         # self.slow = config["slow_action"]
#
#         if config["slow_tp"]:
#             self.tp_range = range(-1, 2)
#         else:
#             self.tp_range = LoRaParameters.TRANSMISSION_POWERS
#
#         if config["slow_sf"]:
#             self.sf_range = range(-1, 2)
#         else:
#             self.sf_range = LoRaParameters.SPREADING_FACTORS
#
#         if config["slow_channel"]:
#             self.channel_range = range(-1, 2)
#         else:
#             self.channel_range = LoRaParameters.DEFAULT_CHANNELS
#
#         for tp in self.tp_range:
#             for sf in self.sf_range:
#                 for ch in self.channel_range:
#                     tensor = torch.tensor([tp, sf, ch])
#                     self.index_to_action.append(tensor)
#                     self.action_to_index[tensor] = curr_idx
#                     curr_idx += 1
#         self.action_size = curr_idx
#
#         self.gamma = config["gamma"]
#         self.epsilon = config["epsilon"]
#         self.alpha = alpha
#         self.losses = {}
#
#
#         tps = len(LoRaParameters.TRANSMISSION_POWERS)
#         sfs = len(LoRaParameters.SPREADING_FACTORS)
#         chs = len(LoRaParameters.DEFAULT_CHANNELS)
#
#         self.q_table = torch.Tensor(tps, sfs, chs, self.action_size)
#
#         # start with epsilon = 1 / 2
#         self.epsilon_counter = 2
#
#         # start with learning rate 1 / 2^1
#         self.alpha_counter = 1
#
#     def convert_state_to_index(self, s):
#         tp_S, sf_S, channel_S = s
#         return (LoRaParameters.TRANSMISSION_POWERS.index(tp_S),
#                 LoRaParameters.SPREADING_FACTORS.index(sf_S),
#                 LoRaParameters.DEFAULT_CHANNELS.index(channel_S))
#
#     def assign_nodes(self, nodes):
#
#         if (len(nodes) == 0):
#             raise Exception("Empty cluster of nodes is being passed to the Learning Agent")
#
#         self.nodes = nodes
#         for node in nodes:
#             node.assign_learning_agent(agent=self)
#
#     def choose_next_action(self, curr_s):
#         # curr_s = self.current_s()
#
#         output = self.q_table[self.convert_state_to_index(curr_s)]
#         # max_action refers to the maximum reward value among the possible actions that can be taken
#         max_action = torch.max(output)
#
#         pi = []
#         max_found = False
#
#         for a in output:
#             if (a == max_action and not max_found):
#                 pi.append(1 - self.epsilon + (self.epsilon / self.action_size))
#                 max_found = True
#             else:
#                 pi.append(self.epsilon / self.action_size)
#
#         if (self.config["GLIE"]):
#             # Update epsilon to ensure convergence according to GLIE
#             self.epsilon_counter += 1
#             self.epsilon = 1 / self.epsilon_counter
#
#         pi[-1] = 1 - sum(pi[:-1])
#
#         if (pi[-1] < 0):
#             print(f"Last value is negative: {pi[-1]}")
#             pi[-1] = 0
#             # TODO catch Value errors for weird pi values and report them
#
#         try:
#             selected_value = np.random.choice(output.detach().numpy(), p=pi)
#         except ValueError:
#             print("Problem with the pi values!")
#             exit(-1)
#
#         if math.isnan(selected_value):
#             # print("NAN ENCOUNTERED")
#             return random.choice(self.index_to_action)
#
#         ret = self.index_to_action[output.detach().numpy().tolist().index(selected_value)]
#         return ret
#
#     # transition = (s, a, r, s')
#     def train_q_network(self, transition):
#         s, a, reward, next_s = transition
#
#         if self.sarsa:
#             next_a = self.choose_next_action(next_s)
#             next_a_index = self.action_to_index[next_a]
#
#         s_index = self.convert_state_to_index(s)
#         a_index = self.action_to_index[a]
#         next_s_index = self.convert_state_to_index(next_s)
#
#         q_s_a = self.q_table[s_index][a_index]
#
#         if math.isnan(q_s_a):
#             q_s_a = 0
#
#         max_q_s_a = torch.max(self.q_table[next_s_index][a_index])
#
#         if (self.sarsa):
#             next_q_s_a = self.q_table[next_s_index][next_a_index]
#             self.q_table[s_index][a_index] = q_s_a + self.alpha * (reward + self.gamma * next_q_s_a - q_s_a)
#         if (self.mc):
#             self.q_table[s_index][a_index] = q_s_a + self.alpha * (reward - q_s_a)
#         else:
#             self.q_table[s_index][a_index] = q_s_a + self.alpha * (reward + self.gamma * max_q_s_a - q_s_a)
#
#         if self.config["Robbins-Monroe"]:
#             self.alpha = 1 / (2 ** self.alpha_counter)
#             self.alpha_counter += 1
#
# class DeepLearningAgent:
#
#     def __init__(self, env, depth, config, lr=0.001):
#
#         self.type = "Deep Q Learning"
#         self.sarsa = config["sarsa"]
#         self.nodes = []
#         self.config = config
#         self.env = env
#
#         # Boolean flags
#         self.double_deep = config["double_deep"]
#         self.replay_buffer = config["replay_buffer"]
#
#         self.index_to_action = []
#         self.action_to_index = {}
#         curr_idx = 0
#
#         # self.slow = config["slow_action"]
#
#         if config["slow_tp"]:
#             self.tp_range = range(-1, 2)
#         else:
#             self.tp_range = LoRaParameters.TRANSMISSION_POWERS
#
#         if config["slow_sf"]:
#             self.sf_range = range(-1, 2)
#         else:
#             self.sf_range = LoRaParameters.SPREADING_FACTORS
#
#         if config["slow_channel"]:
#             self.channel_range = range(-1, 2)
#         else:
#             self.channel_range = LoRaParameters.DEFAULT_CHANNELS
#
#         for tp in self.tp_range:
#             for sf in self.sf_range:
#                 for ch in self.channel_range:
#                     tensor = torch.tensor([tp, sf, ch])
#                     self.index_to_action.append(tensor)
#                     self.action_to_index[tensor] = curr_idx
#                     curr_idx += 1
#         self.action_size = curr_idx
#
#         # Initialize Q-network and Target network.
#         # self.q_network = Network(input_dimensions=len(config["state_space"]), output_dimensions=self.action_size, depth=depth)
#         self.q_network = Network(input_dimensions=len(config["state_space"]), output_dimensions=self.action_size, depth=depth).to(config["device"])
#         # self.target_network = Network(input_dimensions=len(config["state_space"]), output_dimensions=self.action_size, depth=depth)
#         self.target_network = Network(input_dimensions=len(config["state_space"]), output_dimensions=self.action_size, depth=depth).to(config["device"])
#         self.target_network.load_state_dict(self.q_network.state_dict())
#
#         self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)
#
#         self.gamma = config["gamma"]
#         self.epsilon = config["epsilon"]
#         self.lr = lr
#
#         self.losses = {}
#
#         self.buffer = ReplayBuffer(50)
#         self.batch_size = 10
#         self.target_update_counter= 0
#         self.target_update_rate = 10
#         self.device = config["device"]
#         self.config = config
#
#         # start with epsilon = 1 / 2
#         # this counter determines the current epsilon value
#         self.epsilon_value_counter = 2
#         # these counters determine when the epsilon value is updated
#         self.epsilon_update_counter = 0
#         self.epsilon_update_rate = None
#
#         # start with learning rate 1 / 2^1
#         # this counter determines the current alpha value
#         self.alpha_value_counter = 1
#         # these counters determine when the epsilon value is updated
#         self.alpha_update_counter = 0
#         self.alpha_update_rate = None
#
#     def assign_nodes(self, nodes):
#         num_nodes = len(nodes)
#         if (num_nodes == 0):
#             raise Exception("Empty cluster of nodes is being passed to the Learning Agent")
#
#         self.nodes = nodes
#         for node in nodes:
#             node.assign_learning_agent(agent=self)
#
#         self.epsilon_update_rate = num_nodes
#         self.alpha_update_rate = num_nodes
#
#     def choose_next_action(self, curr_s):
#         # curr_s = self.current_s()
#
#         curr_s = curr_s.to(self.device)
#         output = self.q_network(curr_s)
#         output = output.cpu()
#
#         max_action = torch.max(output)
#         pi = []
#         max_found = False
#         for a in output:
#             if (a == max_action and not max_found):
#                 pi.append(1 - self.epsilon + (self.epsilon / self.action_size))
#                 max_found = True
#             else:
#                 pi.append(self.epsilon / self.action_size)
#
#         if (self.config["GLIE"] ):
#             if self.config["slow_epsilon"]:
#                 self.epsilon_update_counter += 1
#                 if self.epsilon_update_counter == self.epsilon_update_rate:
#                     # Update epsilon to ensure convergence according to GLIE
#                     self.epsilon = 1 / self.epsilon_value_counter
#                     self.epsilon_value_counter += 1
#                     self.epsilon_update_counter = 0
#             else:
#                 self.epsilon_value_counter += 1
#                 self.epsilon = 1 / self.epsilon_value_counter
#
#         pi[-1] = 1 - sum(pi[:-1])
#
#         if (pi[-1] < 0):
#             print(f"Last value is negative: {pi[-1]}")
#             pi[-1] = 0
#             # TODO catch Value errors for weird pi values and report them
#
#         try:
#             selected_value = np.random.choice(output.detach().numpy(), p=pi)
#         except ValueError:
#             print("Problem with the pi values!")
#             exit(-1)
#
#         ret = self.index_to_action[output.detach().numpy().tolist().index(selected_value)]
#         return ret
#
#     def train_q_network(self, transition):
#
#         self.optimiser.zero_grad()
#
#         if not self.replay_buffer or self.sarsa:
#
#             loss = self._calculate_loss(transition)
#
#         else:
#             # Implement replay buffer (reminder: cannot implement replay buffer for SARSA!)
#             self.buffer.push(transition)
#
#             if self.buffer.sizeof() < self.batch_size:
#                 return
#
#             transitions, positions = self.buffer.sample(self.batch_size)
#             loss = self._calculate_batch_loss(transitions, positions)
#
#             self.target_update_counter += 1
#
#             if (self.double_deep):
#                 # Synchronize Q and Target Networks every now and then
#                 if self.target_update_counter == self.target_update_rate:
#                     self.target_update_counter = 0
#                     self.target_network.load_state_dict(self.q_network.state_dict())
#
#         loss.backward()
#
#         if len(self.optimiser.param_groups) > 1:
#             raise Exception("Number of parameter groups for the optimizer is more than one")
#
#         if (self.config["Robbins-Monroe"]):
#             if (self.config["slow_alpha"]):
#                 self.alpha_update_counter += 1
#                 if (self.alpha_update_counter == self.alpha_update_rate):
#                     self.lr = 1 / (2 ** self.alpha_value_counter)
#                     self.optimiser.param_groups[0]['lr'] = self.lr
#                     self.alpha_value_counter += 1
#                     self.alpha_update_counter = 0
#             else:
#                 self.lr = 1 / (2 ** self.alpha_value_counter)
#                 self.optimiser.param_groups[0]['lr'] = self.lr
#                 self.alpha_value_counter += 1
#
#         self.optimiser.step()
#
#         return loss.item()
#
#     def _calculate_batch_loss(self, transitions, positions):
#         # transition — (s, a, r, s')
#
#         transposed = list(zip(*transitions))
#
#         states = torch.cat([s.unsqueeze(0) for s in transposed[0]])
#         actions = torch.cat(
#             [torch.tensor([self.action_to_index[a]], dtype=torch.int64).unsqueeze(0) for a in transposed[1]])
#         rewards = torch.cat([torch.tensor([r]).unsqueeze(0) for r in transposed[2]])
#
#         next_states = torch.cat([ns.unsqueeze(0) for ns in transposed[3]])
#
#         if self.sarsa:
#             ### TODO This line is too complicated, need to implement choose_next_action for batch input ###
#             raise Exception("Shouldn't reach this code! Cannot implement Replay Buffer for SARSA!")
#             next_actions = torch.cat(
#                 [torch.tensor([self.action_to_index[self.choose_next_action(ns)]], dtype=torch.int64).unsqueeze(0) for
#                  ns in transposed[3]])
#             target = rewards + self.gamma * self.q_network(next_states).gather(1, next_actions)
#         else:
#             ### the same operation broken down ###
#             # temp = self.q_network(next_states)
#             # temp = torch.max(temp, dim=1)[0]
#             # temp = temp.detach()
#             # temp = temp.squeeze(0).view(self.batch_size, 1)
#
#             if (self.double_deep):
#
#                 # Simple implementation of Target Network optimization
#                 # temp = torch.max(self.target_network(next_states), dim=1)[0].detach().squeeze(0).view(self.batch_size, 1)
#                 temp = self.target_network(next_states).cpu()
#                 optimal_action_indexes = torch.argmax(temp, dim=1)
#
#                 temp = self.q_network(next_states).cpu()
#                 temp = torch.tensor([q[i] for (q, i) in zip(temp, optimal_action_indexes)]).squeeze(0).view(self.batch_size, 1)
#
#             else:
#
#                 temp = self.q_network(next_states).cpu()
#                 temp = torch.max(temp, dim=1)[0].detach().squeeze(0).view(self.batch_size, 1)
#
#             target = rewards + temp * self.gamma
#
#
#         predicted = self.q_network(states).cpu()
#         predicted = predicted.gather(1, actions)
#
#         # Commenting this code out, attempt to use squared loss value instead of absolute difference
#         # relative to each other the values should be the same (abs() and pow(2) are similar operations)
#         # delta = target - predicted
#         # loss = delta.pow(2)
#
#         # reduction="None" ensures loss is calculated across the whole batch
#         # 10 losses are produced (in one tensor of size [10, 1])
#         loss = torch.nn.MSELoss(reduction="none")
#         loss = loss(target.float(), predicted.float())
#
#         weights = loss + 1e-5
#         weights = weights.squeeze(1).tolist()
#         self.buffer.update_weights(positions, weights)
#
#         loss = loss.mean()
#         self.losses[self.env.now] = loss
#
#         return loss
#
#     # transition = (s, a, r, s')
#     def _calculate_loss(self, transition):
#         s, a, reward, next_s = transition
#         if self.sarsa:
#             next_a = self.choose_next_action(next_s)
#             target = reward + self.gamma * self.q_network(next_s)[self.action_to_index[next_a]]
#         else:
#             if (self.double_deep):
#                 temp = self.target_network(next_s).cpu()
#                 target = reward + self.gamma * torch.max(temp)
#             else:
#                 temp = self.q_network(next_s).cpu()
#                 target = reward + self.gamma * torch.max(temp)
#
#         # is_it = a in self.index_to_action
#         # print(is_it)
#         predicted = self.q_network(s)
#         # predicted = self.q_network(s).cpu()
#         predicted = predicted[self.action_to_index[a]]
#         loss = torch.nn.MSELoss()
#         loss = loss(target, predicted)
#         self.losses[self.env.now] = loss
#         return loss

# old batch loss for DeepLearningAgent
# def _calculate_batch_loss_old(self, transitions):
    #     # transition — (s, a, r, s')
    #
    #     transposed = list(zip(*transitions))
    #
    #     states = torch.cat([s.unsqueeze(0) for s in transposed[0]])
    #     actions = torch.cat([torch.tensor([self.action_to_index[a]], dtype=torch.int64).unsqueeze(0) for a in transposed[1]])
    #     rewards = torch.cat([torch.tensor([r]).unsqueeze(0) for r in transposed[2]])
    #
    #     next_states = torch.cat([ns.unsqueeze(0) for ns in transposed[3]])
    #
    #     if self.sarsa:
    #         ### TODO This line is too complicated, need to implement choose_next_action for batch input ###
    #         raise Exception("Shouldn't reach this code! Cannot implement Replay Buffer for SARSA!")
    #         next_actions = torch.cat([torch.tensor([self.action_to_index[self.choose_next_action(ns)]], dtype=torch.int64).unsqueeze(0) for ns in transposed[3]])
    #         target = rewards + self.gamma * self.q_network(next_states).gather(1, next_actions)
    #     else:
    #         ### the same operation broken down ###
    #         # temp = self.q_network(next_states)
    #         # temp = torch.max(temp, dim=1)[0]
    #         # temp = temp.detach()
    #         # temp = temp.squeeze(0).view(self.batch_size, 1)
    #         temp = torch.max(self.q_network(next_states), dim=1)[0].detach().squeeze(0).view(self.batch_size, 1)
    #         target = rewards + temp * self.gamma
    #
    #     predicted = self.q_network(states)
    #     predicted = predicted.gather(1, actions)
    #
    #     loss = torch.nn.MSELoss()
    #     loss = loss(target, predicted)
    #
    #     self.losses[self.env.now] = loss
    #
    #     return loss

### END ###
