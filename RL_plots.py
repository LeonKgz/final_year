import matplotlib.pyplot as plt
import numpy as np

from Global import Config
from clustering_nodes import search_closest


class RL_plots:

    def __init__(self, env):
        # self.rewards_records = {}
        self.env = env

    # def register_reward(self, node_id, reward):
    #     if (node_id not in self.rewards_records):
    #         self.rewards_records[node_id] = {'rewards': [], 'time': []}
    #     self.rewards_records[node_id]['time'].append(self.env.now)
    #     self.rewards_records[node_id]['rewards'].append(reward)

    def plot_merge_for_nodes(self, nodes, measurement, condition, title=""):
        plt.figure()

        len_acc = 0
        for node in nodes:
            len_acc += len(list(node.rl_measurements[measurement].keys()))
        avg_len = int(len_acc / len(nodes))

        times = [i for i in range(0, Config.SIMULATION_TIME, int(Config.SIMULATION_TIME / avg_len))]
        reward_records = {}
        for time in times:
            reward_records[time] = []

        for node in nodes:
            for t, r in node.rl_measurements[measurement].items():
                time_slot = search_closest(times, t)
                reward_records[time_slot].append(r)

        for t, rs in reward_records.items():

            if (condition == "average"):
                reward_records[t] = np.mean(reward_records[t])
            elif (condition == "sum"):
                reward_records[t] = np.sum(reward_records[t])
            else:
                raise Exception("Condition is not valid!")

        plt.plot(list(reward_records.keys()), list(reward_records.values()))

        plt.xlabel("time")
        plt.ylabel(f"{condition} {measurement} over all nodes")
        plt.title(title)
        plt.show()

        # common_rewards = []
        # common_time = []
        # common_pairs = {}
        #
        # for id, records in rewards_records.items():
        #     for (time, reward) in zip(records['time'], records['rewards']):
        #         if (time not in common_time):
        #             common_pairs[time] = reward
        #             common_time.append(time)
        #         else:
        #             common_pairs[time] = common_pairs[time] + reward / 2
        #
        # common_time.sort()
        #
        # for t in common_time:
        #     common_rewards.append(common_pairs[t])

        # plt.plot(common_time, common_rewards)
        # plt.show()

        # for (id, records) in self.rewards_records.items():
        #     plt.plot(records['time'], records['rewards'], label=id)
        # plt.xlabel("time")
        # plt.ylabel("Reward Score")
        # plt.show()

    def plot_losses_for_agents(self, agents):

        plt.figure()
        agent_id = 1
        for agent in agents:
            losses = agent.losses
            plt.plot(list(losses.keys()), list(losses.values()), label=agent_id)
            agent_id += 1

        plt.xlabel("time")
        plt.ylabel("Loss")
        plt.show()

    def plot_rewards_and_losses(self, nodes, agents, depth=None, lr=None, epsilon=None, gamma=None, show=True, save=False):

        f, axarr = plt.subplots(2, figsize=(7, 7))

        ax_r = axarr[0]
        ax_l = axarr[1]

        # Plotting rewards
        for node in nodes:
            rewards = node.rl_measurements["rewards"]
            ax_r.plot(list(rewards.keys()), list(rewards.values()), label=node.id)

        ax_r.set_xlabel("Time")
        ax_r.set_ylabel("Reward per Node")

        # Plotting losses
        agent_id = 1
        for agent in agents:
            losses = agent.losses
            plt.plot(list(losses.keys()), list(losses.values()), label=agent_id)
            agent_id += 1

        ax_l.set_xlabel("Time")
        ax_l.set_ylabel("Loss per Agent")

        ax_r.set_title(f"depth={depth}_lr={lr}_epsilon={epsilon}_gamma={gamma}_.png")

        if save:
            f.savefig(f"RL_plots/depth={depth}_lr={lr}_epsilon={epsilon}_gamma={gamma}_.png")

        if show:
            plt.show()

