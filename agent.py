from LoRaParameters import LoRaParameters

import numpy as np
import torch

# Reinforcement Learning Agent section
class LearningAgent:

    def __init__(self):

        self.nodes = None

        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimensions=8, output_dimensions=90)
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

        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.action_size = 90
        self.gamma = 0.1
        self.epsilon = 0.5

    def assign_nodes(self, nodes):
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

    # def take_action(self, a):
    #     map(lambda node: node.take_action(a), self.nodes)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    # transition = (s, a, r, s')
    def _calculate_loss(self, transition):
        s, a, reward, next_s = transition
        target = reward + self.gamma * torch.max(self.q_network(next_s))
        # is_it = a in self.index_to_action
        # print(is_it)
        predicted = self.q_network(s)
        predicted = predicted[self.action_to_index[a]]
        loss = torch.nn.MSELoss()
        return loss(target, predicted)

class Network(torch.nn.Module):

    def __init__(self, input_dimensions, output_dimensions):
        super(Network, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dimensions, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=output_dimensions)
        )

    def forward(self, input):
        return self.network(input)
