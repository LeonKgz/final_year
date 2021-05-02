import datetime

import matplotlib.pyplot as plt
import numpy as np
import simpy
import torch
import pandas as pd
import pickle

import PropagationModel
from AirInterface import AirInterface
from EnergyProfile import EnergyProfile
from Gateway import Gateway
from Global import Config
from LoRaParameters import LoRaParameters
from Location import Location
from Node import Node
from RL_plots import RL_plots
from SINRModel import SINRModel
from SNRModel import SNRModel
from agent import DeepLearningAgent, LearningAgent
from clustering_nodes import cluster_nodes, search_closest


def plot_time(_env, sim_time):
    while True:
        print('.', end='', flush=True)
        yield _env.timeout(np.round(sim_time / 10))

def run_simulation(conf, adr, training, num_nodes):

    energy_per_bit = 0
    tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}  # measured TX power for each possible TP
    middle = np.round(Config.CELL_SIZE / 2)
    gateway_location = Location(x=middle, y=middle, indoor=False)
    # plt.scatter(middle, middle, color='red')
    env = simpy.Environment()
    gateway = Gateway(env, gateway_location)
    nodes = []
    air_interface = AirInterface(gateway, PropagationModel.LogShadow(), SNRModel(), SINRModel(), env)
    rl_plots = RL_plots(env=env)

    for node_id in range(num_nodes):
        location = Location(min=0, max=Config.CELL_SIZE, indoor=False)
        # location = Location(x=60, y=60, indoor=True)
        # TODO check if random location is more than 1m from gateway
        # node = Node(id, EnergyProfile())
        # energy_profile = EnergyProfile(5.7e-3, 15, tx_power_mW,
        #                                rx_power={'pre_mW': 8.2, 'pre_ms': 3.4, 'rx_lna_on_mW': 39, 'rx_lna_off_mW': 34,
        #                                          'post_mW': 8.3, 'post_ms': 10.7})
        # lora_param = LoRaParameters(freq=np.random.choice(LoRaParameters.DEFAULT_CHANNELS),
        #                             sf=np.random.choice(LoRaParameters.SPREADING_FACTORS),
        #                             bw=125, cr=5, crc_enabled=1, de_enabled=0, header_implicit_mode=0, tp=14)
        # # lora_param = LoRaParameters(freq=np.random.choice(LoRaParameters.DEFAULT_CHANNELS),
        # #                             sf=12,
        # #                             bw=125, cr=5, crc_enabled=1, de_enabled=0, header_implicit_mode=0, tp=14)
        # node = Node(node_id, energy_profile, lora_param, 1000 * 60*60, process_time=5, adr=False, location=location,
        #             base_station=gateway, env=env, payload_size=16, air_interface=air_interface)

        payload_size = 25
        transmission_rate = 0.02e-3  # 12*8 bits per hour (1 typical packet per hour)


        energy_profile = EnergyProfile(5.7e-3, 15, tx_power_mW,
                                       rx_power={'pre_mW': 8.25, 'pre_ms': 3.4, 'rx_lna_on_mW': 36.96,
                                                 'rx_lna_off_mW': 34.65,
                                                 'post_mW': 8.3, 'post_ms': 10.7})
        lora_param = LoRaParameters(freq=np.random.choice(LoRaParameters.DEFAULT_CHANNELS),
                                    sf=np.random.choice(LoRaParameters.SPREADING_FACTORS),
                                    bw=125, cr=5, crc_enabled=1, de_enabled=0, header_implicit_mode=0, tp=14)

        node = Node(node_id, energy_profile, lora_param, sleep_time=(8 * payload_size / transmission_rate),
                    process_time=5,
                    location=location,
                    adr=adr,
                    confirmed_messages=conf,
                    training=training,
                    base_station=gateway, env=env, payload_size=payload_size, air_interface=air_interface, reward="normal")

        nodes.append(node)
        plt.scatter(location.x, location.y, color='blue')


    axes = plt.gca()
    axes.set_xlim([0, Config.CELL_SIZE])
    axes.set_ylim([0, Config.CELL_SIZE])
    # plt.show()

    # Creation of clusters and assignment of one learning agent per cluster
    print("\nCreating clusters of nodes...")
    sector_size = 100
    clusters = cluster_nodes(nodes, sector_size=sector_size)
    print("Finished creating clusters\n")

    # switch = True
    # for cluster in clusters.keys():
    #     if switch:
    #         color = 'red'
    #     else:
    #         color = 'blue'
    #     switch = not switch
    #     for node in clusters[cluster]:
    #         plt.scatter(node.location.x, node.location.y, color=color)
    #
    # plt.show()

    for cluster in clusters.keys():
        for node in clusters[cluster]:
            plt.scatter(node.location.x, node.location.y, color='blue')

    for node in clusters[list(clusters.keys())[13]]:
        plt.scatter(node.location.x, node.location.y, color='red')

    major_ticks = np.arange(0, 1001, sector_size)

    axes.set_xticks(major_ticks)
    axes.set_yticks(major_ticks)

    plt.grid(True)
    plt.show()

    agents = []
    for (cluster_center_location, cluster) in clusters.items():
        agent = DeepLearningAgent(env=env, depth=4, gamma=0.9, epsilon=0.9)
        # making sure each agent is assigned to at least one node
        # TODO get rid of empty clusters earlier (problem of uneven distribution of nodes!)
        if len(cluster) > 0:
            agent.assign_nodes(cluster)
            agents.append(agent)

    # The simulation is kicked off after agents are assigned to respective clusters
    for node in nodes:
        env.process(node.run())

    env.process(plot_time(env))

    d = datetime.timedelta(milliseconds=Config.SIMULATION_TIME)
    print('Running simulator for {}.'.format(d))
    env.run(until=Config.SIMULATION_TIME)

    # rl_plots.plot_rewards_for_nodes(nodes=nodes)
    # rl_plots.plot_losses_for_agents(agents=agents)

    rl_plots.plot_rewards_and_losses(nodes=nodes, agents=agents)
    rl_plots.plot_merge_for_nodes(nodes=nodes, measurement="rewards", condition="sum", title="")
    rl_plots.plot_merge_for_nodes(nodes=nodes, measurement="rewards", condition="sum", title="")
    # plt.figure()
    # for node in nodes:
    #     # node.log()
    #     # measurements = air_interface.get_prop_measurements(node.id)
    #     # node.plot(measurements)
    #     node.plot_rewards()
    #     # break
    #     # energy_per_bit += node.energy_per_bit()
    #     # print('E/bit {}'.format(energy_per_bit))
    #     # print(f"Node {node.id}")
    # plt.show()

    energy_per_bit = energy_per_bit/1000.0
    print('E/bit {}'.format(energy_per_bit))
    gateway.log()
    air_interface.log()
    # air_interface.plot_packets_in_air()

def init_nodes(conf, adr, training, num_nodes, reward, locations, agent_to_nodes=None, sector_size=100, deep=False, sarsa=False):
    energy_per_bit = 0
    tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}  # measured TX power for each possible TP
    middle = np.round(Config.CELL_SIZE / 2)
    gateway_location = Location(x=middle, y=middle, indoor=False)
    # plt.scatter(middle, middle, color='red')
    env = simpy.Environment()
    gateway = Gateway(env, gateway_location)
    nodes = []
    air_interface = AirInterface(gateway, PropagationModel.LogShadow(), SNRModel(), SINRModel(), env)

    if (agent_to_nodes != None):
        locations = []
        for locs in list(agent_to_nodes.values()):
            locations += [Location(x=x, y=y, indoor=False) for (x, y) in locs]

    location_to_node = {}

    for node_id in range(num_nodes):
        # location = Location(min=0, max=Config.CELL_SIZE, indoor=False)
        location = locations[node_id]
        payload_size = 25
        transmission_rate = 0.02e-3  # 12*8 bits per hour (1 typical packet per hour)

        energy_profile = EnergyProfile(5.7e-3, 15, tx_power_mW,
                                       rx_power={'pre_mW': 8.25, 'pre_ms': 3.4, 'rx_lna_on_mW': 36.96,
                                                 'rx_lna_off_mW': 34.65,
                                                 'post_mW': 8.3, 'post_ms': 10.7})
        lora_param = LoRaParameters(freq=np.random.choice(LoRaParameters.DEFAULT_CHANNELS),
                                    sf=np.random.choice(LoRaParameters.SPREADING_FACTORS),
                                    bw=125, cr=5, crc_enabled=1, de_enabled=0, header_implicit_mode=0, tp=14)

        node = Node(node_id, energy_profile, lora_param, sleep_time=(8 * payload_size / transmission_rate),
                    process_time=5,
                    location=location,
                    adr=adr,
                    confirmed_messages=conf,
                    training=training,
                    base_station=gateway, env=env, payload_size=payload_size, air_interface=air_interface,
                    reward=reward)

        nodes.append(node)
        location_to_node[(location.x, location.y)] = node

    agents = []

    if (agent_to_nodes == None):
        # Creation of clusters and assignment of one learning agent per cluster
        print("Creating clusters of nodes...")
        clusters = cluster_nodes(nodes, sector_size=sector_size)
        print("Finished creating clusters")

        for (cluster_center_location, cluster) in clusters.items():
            agent = DeepLearningAgent(env=env, depth=4, gamma=0.5, epsilon=0.9, lr=0.001, sarsa=sarsa) if deep \
                else LearningAgent(env=env, gamma=0.5, epsilon=0.9, alpha=0.5, sarsa=sarsa)

            # making sure each agent is assigned to at least one node
            # TODO (problem of uneven distribution of nodes!)
            # if len(cluster) > 0:
            agent.assign_nodes(cluster)
            agents.append(agent)
    else:

        for id in list(agent_to_nodes.keys()):

            agent = DeepLearningAgent(env=env, depth=4, gamma=0.5, epsilon=0.9, lr=0.001)
            agent.q_network.load_state_dict(torch.load(f"./model/agent_{id}.pth"))
            agent.q_network.eval()
            agents.append(agent)

            cluster = []
            for location in agent_to_nodes[id]:
                cluster.append(location_to_node[location])

            agent.assign_nodes(cluster)

    return nodes, agents, env

# def assign_agents(nodes, clusters, env):
#
#     agents = []
#     for (cluster_center_location, cluster) in clusters.items():
#         agent = LearningAgent(env=env, depth=4, gamma=0.9, epsilon=0.9, lr=0.001)
#         # making sure each agent is assigned to at least one node
#         # TODO get rid of empty clusters earlier (problem of uneven distribution of nodes!)
#         if len(cluster) > 0:
#             agent.assign_nodes(cluster)
#             agents.append(agent)

def run_nodes(nodes, env, days):
    # The simulation is kicked off after agents are assigned to respective clusters
    for node in nodes:
        env.process(node.run())
    sim_time = 1000*60*60*24*days
    d = datetime.timedelta(milliseconds=sim_time)
    print('Running simulator for {}.'.format(d))
    env.process(plot_time(env, sim_time))
    env.run(until=sim_time)

import os
import glob

def save_agents(agents):

    files = glob.glob('./model/*')
    for f in files:
        os.remove(f)

    agent_to_node = {}

    id = 0
    for agent in agents:
        torch.save(agent.q_network.state_dict(), f"./model/agent_{id}.pth")
        agent_to_node[id] = [(node.location.x, node.location.y) for node in agent.nodes]
        id += 1

    with open('./model/agent_to_node', 'wb') as file:
        pickle.dump(agent_to_node, file, protocol=pickle.HIGHEST_PROTOCOL)

    # pd_dict = pd.DataFrame.from_dict(agent_to_node)
    # pd_dict.to_pickle("./model/agent_to_node")

# loads a dicitonary mapping agent model ids to locations of nodes (in a cluster)
# associated with that model (learning agent)
def load_agents(clusters="./model/agent_to_node"):

    with open('./model/agent_to_node', 'rb') as file:
        ret = pickle.load(file)
        return ret

    # return pd.read_pickle(clusters)

def compare_before_and_after(title):

    num_nodes = 1000

    locations = list()
    for i in range(num_nodes):
        locations.append(Location(min=0, max=Config.CELL_SIZE, indoor=False))

    configurations = [
        # {
        #     "title": "NO ADR NO CONF",
        #     "conf": False,
        #     "adr": False,
        #     "training": False,
        #     "deep": False,
        #     "load": False,
        #     "save": False,
        #     "num_nodes": num_nodes,
        #     "reward": title,
        #     "days": 10,
        #     "locations": locations},
        # {
        #     "title": "ADR CONF",
        #     "conf": True,
        #     "adr": True,
        #     "training": False,
        #     "num_nodes": num_nodes,
        #     "reward": title,
        #     "days": 10,
        #     "locations": locations},
        {
            "title": "Q learning",
            "conf": False,
            "adr": False,
            "training": True,
            "deep": False,
            "sarsa": False,
            "load": False,
            "save": False,
            "num_nodes": num_nodes,
            "reward": title,
            "days": 1,
            "locations": locations},
        {
            "title": "SARSA",
            "conf": False,
            "adr": False,
            "training": True,
            "deep": False,
            "sarsa": True,
            "load": False,
            "save": False,
            "num_nodes": num_nodes,
            "reward": title,
            "days": 1,
            "locations": locations},
        {
            "title": "SARSA",
            "conf": False,
            "adr": False,
            "training": True,
            "deep": False,
            "sarsa": True,
            "load": False,
            "save": False,
            "num_nodes": num_nodes,
            "reward": title,
            "days": 1,
            "locations": locations},
        {
            "title": "SARSA",
            "conf": False,
            "adr": False,
            "training": True,
            "deep": False,
            "sarsa": True,
            "load": False,
            "save": False,
            "num_nodes": num_nodes,
            "reward": title,
            "days": 1,
            "locations": locations},
        # {
        #     "title": "Deep Q Learning",
        #     "conf": False,
        #     "adr": False,
        #     "training": True,
        #     "deep": True,
        #     "sarsa": False,
        #     "load": False,
        #     "save": False,
        #     "num_nodes": num_nodes,
        #     "reward": title,
        #     "days": 10,
        #     "locations": locations},
        # {
        #     "title": "Deep SARSA",
        #     "conf": False,
        #     "adr": False,
        #     "training": True,
        #     "deep": True,
        #     "sarsa": True,
        #     "load": False,
        #     "save": False,
        #     "num_nodes": num_nodes,
        #     "reward": title,
        #     "days": 10,
        #     "locations": locations},
    ]

    first_run = True
    f, axarr = None, None

    for config_cnt, config in enumerate(configurations):
        simulation_time = config["days"]*1000*60*60*24
        nodes, agents, env = init_nodes(conf=config["conf"],
                                adr=config["adr"],
                                training=config["training"],
                                num_nodes=config["num_nodes"],
                                reward=config["reward"],
                                locations=config["locations"],
                                agent_to_nodes=load_agents() if config["load"] else None,
                                deep=config["deep"],
                                sarsa=config["sarsa"])

        run_nodes(nodes, env, days=config["days"])

        if config["save"]:
            save_agents(agents)

        num_categories = len(list(nodes[0].rl_measurements.keys()))

        if first_run:
            f, axarr = plt.subplots((num_categories * 2) + 2, len(configurations), figsize=(30, 30), sharex='col', sharey='row')
        first_run = False

        length_per_parameter = {}
        times_per_parameter = {}
        # sum_per_parameter = {}
        avg_per_parameter = {}

        for parameter in list(nodes[0].rl_measurements.keys()):
            times_per_parameter[parameter] = []
            length_per_parameter[parameter] = 0
            # sum_per_parameter[parameter] = {}
            avg_per_parameter[parameter] = {}

        energy_avg = 0
        num_packets_avg = 0

        # Display information about current configuration (at the top of all the plots)
        props = dict(alpha=0.6)

        simulation_info = "FLAVOUR — {}\n" \
                          "NUMBER OF NODES — {}\n" \
                          "SIMULATION TIME — {} DAYS\n" \
                          "GAMMA — {} DAYS\n" \
                          "EPSILON — {} DAYS\n" \
            .format(
                config["title"],
                config["num_nodes"],
                config["days"],
                agents[0].gamma,
                agents[0].epsilon
            )

        if (config["deep"]):
            simulation_info += "LEARNING RATE — {}".format(agents[0].lr)
        else:
            simulation_info += "ALPHA — {}".format(agents[0].alpha)

        if (len(configurations) == 1):
            axarr[0].text(0.1, 0.9, simulation_info, fontsize=20, bbox=props, verticalalignment='top', transform=axarr[0].transAxes)
        else:
            axarr[0][config_cnt].text(0.1, 0.9, simulation_info, fontsize=20, bbox=props, verticalalignment='top', transform=axarr[0][config_cnt].transAxes)



        for node in nodes:

            energy_avg += node.total_energy_consumed()
            num_packets_avg += node.num_unique_packets_sent

            for parameter_cnt, parameter in enumerate(list(node.rl_measurements.keys())):
                data_dict = node.rl_measurements[parameter]
                time = list(data_dict.keys())
                vals = list(data_dict.values())
                if (len(configurations) == 1):
                    axarr[parameter_cnt * 2 + 1].plot(time, vals)
                    axarr[parameter_cnt * 2 + 1].set_xlabel("Time")
                    axarr[parameter_cnt * 2 + 1].set_ylabel(f"{parameter} per Node")
                    axarr[parameter_cnt * 2 + 1].set_title(config["title"])
                else:
                    axarr[parameter_cnt * 2 + 1][config_cnt].plot(time, vals)
                    axarr[parameter_cnt * 2 + 1][config_cnt].set_xlabel("Time")
                    axarr[parameter_cnt * 2 + 1][config_cnt].set_ylabel(f"{parameter} per Node")
                    axarr[parameter_cnt * 2 + 1][config_cnt].set_title(config["title"])

                length_per_parameter[parameter] += len(time)

        energy_avg = (energy_avg / num_nodes) / 1000
        num_packets_avg = num_packets_avg / num_nodes

        for parameter_cnt, parameter in enumerate(list(nodes[0].rl_measurements.keys())):
            length_per_parameter[parameter] = int(length_per_parameter[parameter] / len(nodes))
            times_per_parameter[parameter] = [i for i in range(0, simulation_time, int(simulation_time / length_per_parameter[parameter]))]

            parameter_records = {}
            for time in times_per_parameter[parameter]:
                parameter_records[time] = []


            for node in nodes:
                for t, v in node.rl_measurements[parameter].items():
                    time_slot = search_closest(times_per_parameter[parameter], t)
                    parameter_records[time_slot].append(v)

            for t, vs in parameter_records.items():
                # sum_per_parameter[parameter][t] = np.sum(vs)
                avg_per_parameter[parameter][t] = np.mean(vs)

            # temp = (parameter_cnt * 3) + 1
            # axarr[temp][config_cnt].plot(sum_per_parameter[parameter].keys(), sum_per_parameter[parameter].values())
            # axarr[temp][config_cnt].set_xlabel("Time")
            # axarr[temp][config_cnt].set_ylabel(f"{parameter} SUM per Node")
            # axarr[temp][config_cnt].set_title(config["title"])

            temp = (parameter_cnt * 2) + 1

            if (len(configurations) == 1):
                axarr[temp].plot(avg_per_parameter[parameter].keys(), avg_per_parameter[parameter].values())
                axarr[temp].set_xlabel("Time")
                axarr[temp].set_ylabel(f"{parameter} AVERAGE per Node")
                axarr[temp].set_title(config["title"])
            else:
                axarr[temp][config_cnt].plot(avg_per_parameter[parameter].keys(), avg_per_parameter[parameter].values())
                axarr[temp][config_cnt].set_xlabel("Time")
                axarr[temp][config_cnt].set_ylabel(f"{parameter} AVERAGE per Node")
                axarr[temp][config_cnt].set_title(config["title"])

        agent_id = 0
        for agent in agents:
            losses = agent.losses

            if (len(configurations) == 1):
                axarr[-1].plot(list(losses.keys()), list(losses.values()), label=agent_id)
                agent_id += 1
                axarr[-1].set_xlabel("Time")
                axarr[-1].set_ylabel("Loss per Agent")
                axarr[-1].set_title(config["title"])
            else:
                axarr[-1][config_cnt].plot(list(losses.keys()), list(losses.values()), label=agent_id)
                agent_id += 1
                axarr[-1][config_cnt].set_xlabel("Time")
                axarr[-1][config_cnt].set_ylabel("Loss per Agent")
                axarr[-1][config_cnt].set_title(config["title"])

        ################################### CALCULATING ENERGY EFFICIENCY AT THE END ##########################

        # temp = []
        # for node in nodes:
        #     rewards = node.rl_measurements["rewards"]
        #     temp.append(list(rewards.values())[-1])
        #     # mean_adr_conf = np.mean(list(rewards.values()))
        # mean = np.mean(temp)
        # print(f"\n Mean throughput (at the end of the simulation) for " + str(config["title"]) + " is " + str(mean))
        # print(f"\n Energy Efficiency (Packets/Joule) = {num_packets_avg / energy_avg}")

        ################################### VISUALIZING ACTION EXPLORATION ##########################

        # if (len(nodes[0].actions) > 0):
        #
        #     curr_cnt = 0
        #
        #     for j in range(0, len(nodes[0].actions)-10, 10):
        #         fig = plt.figure()
        #         ax = fig.add_subplot(projection="3d")
        #         slice = nodes[0].actions[j:j+10]
        #         actions_unzipped = list(zip(*slice))
        #
        #         tps = actions_unzipped[0]
        #         sfs = actions_unzipped[1]
        #         chs = actions_unzipped[2]
        #
        #         for i in range(len(slice)):
        #             ax.text(tps[i], sfs[i], chs[i], f"{curr_cnt}")
        #
        #             print(f"({tps[i]}, {sfs[i]}, {chs[i]})")
        #
        #             curr_cnt += 1
        #             # ax.scatter(xs[i], ys[i], zs[i], marker='*')
        #
        #         ax.plot(tps, sfs, chs, linestyle="-")
        #
        #         # ax.scatter(, , , ftm='o')
        #         ax.set_xlabel("transmission power")
        #         ax.set_ylabel("spreading factor")
        #         ax.set_zlabel("channels")
        #         title = config["title"]
        #         ax.set_title(title)
        #
        #         plt.show()
        #         plt.close()

    plt.show()

def compare_t_b_and_a(title="new"):

    num_nodes = 1000
    num_samples = 10

    locations = list()
    for i in range(num_nodes):
        locations.append(Location(min=0, max=Config.CELL_SIZE, indoor=False))

    nodes_before, env_before = init_nodes(conf=False, adr=False, training=False,
                                          num_nodes=num_nodes, reward=title, locations=locations)
    nodes_rl, env_rl = init_nodes(conf=False, adr=False, training=True,
                                  num_nodes=num_nodes, reward=title, locations=locations)
    nodes_adr_conf, env_adr_conf = init_nodes(conf=True, adr=True, training=False,
                                              num_nodes=num_nodes, reward=title, locations=locations)

    run_nodes(nodes_before, env_before, days=10)
    run_nodes(nodes_rl, env_rl, days=10)
    run_nodes(nodes_adr_conf, env_adr_conf, days=10)

    f, axarr = plt.subplots(num_samples, 3, figsize=(6, 20), sharey=True)

    # for i in range(num_samples):
    #
    #     ax_before = axarr[i][0]
    #     ax_condf_adr = axarr[i][1]
    #     ax_rl = axarr[i][2]

    mean_before = None
    mean_adr_conf = None
    mean_rl = None

    temp = []
    for node in (nodes_before):
        rewards = node.rewards
        temp.append(list(rewards.values())[-1])
        # mean_before = np.mean(list(rewards.values()))
    mean_before = np.mean(temp)

    temp = []
    for node in (nodes_adr_conf):
        rewards = node.rewards
        temp.append(list(rewards.values())[-1])
        # mean_adr_conf = np.mean(list(rewards.values()))
    mean_adr_conf = np.mean(temp)

    temp = []
    for node in (nodes_rl):
        rewards = node.rewards
        temp.append(list(rewards.values())[-1])
        # mean_rl = np.mean(list(rewards.values()))
    mean_rl = np.mean(temp)

    print(f"mean of throughput before - {mean_before}")
    print(f"mean of throughput adr_conf - {mean_adr_conf}")
    print(f"mean of throughput rl - {mean_rl}")

    for i, node in enumerate(nodes_before[:num_samples]):
        ax_before = axarr[i][0]
        rewards = node.rewards
        ax_before.plot(list(rewards.keys()), list(rewards.values()), label=node.id)
        ax_before.set_xlabel("Time")
        ax_before.set_ylabel(f"{title} per Node")
        ax_before.set_title("Before")

    for i, node in enumerate(nodes_adr_conf[:num_samples]):
        ax_adr_conf = axarr[i][1]
        rewards = node.rewards
        ax_adr_conf.plot(list(rewards.keys()), list(rewards.values()), label=node.id)
        ax_adr_conf.set_xlabel("Time")
        ax_adr_conf.set_ylabel(f"{title} per Node")
        ax_adr_conf.set_title("ADR CONF")

    for i, node in enumerate(nodes_rl[:num_samples]):
        ax_rl = axarr[i][2]
        rewards = node.rewards
        ax_rl.plot(list(rewards.keys()), list(rewards.values()), label=node.id)
        ax_rl.set_xlabel("Time")
        ax_rl.set_ylabel(f"{title} per Node")
        ax_rl.set_title("RL")

    plt.show()

def compare_thrughput_before_and_after_for_all_nodes(title="throughput"):

    locations = list()
    for i in range(1000):
        locations.append(Location(min=0, max=Config.CELL_SIZE, indoor=False))

    nodes_before, env_before = init_nodes(conf=False, adr=False, training=False, num_nodes=1000, reward=title, locations=locations)
    nodes_adr_conf, env_adr_conf = init_nodes(conf=True, adr=True, training=True, num_nodes=1000, reward=title, locations=locations)
    nodes_after, env_after = init_nodes(conf=False, adr=False, training=True, num_nodes=1000, reward=title, locations=locations)

    run_nodes(nodes_after, env_after, days=10)
    run_nodes(nodes_before, env_before, days=10)
    run_nodes(nodes_adr_conf, env_adr_conf, days=10)

    f, axarr = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

    ax_before = axarr[0]
    ax_adr_conf = axarr[1]
    ax_after = axarr[2]

    ids_before = []
    throughputs_before = []

    for node in nodes_before:
        ids_before.append(node.id)
        throughputs_before.append(node.get_mean_throughput())
    ax_before.plot(ids_before, throughputs_before)

    ax_before.set_xlabel("Node ID")
    ax_before.set_ylabel("mean throughput per node")
    ax_before.set_title("Before")

    ids_after = []
    throughputs_after = []

    for node in nodes_after:
        ids_after.append(node.id)
        throughputs_after.append(node.get_mean_throughput())
    ax_after.plot(ids_after, throughputs_after)

    ax_after.set_xlabel("Node ID")
    ax_after.set_ylabel("mean throughput per node")
    ax_after.set_title("After (RL)")

    ids_adr_conf = []
    throughputs_adr_conf = []

    for node in nodes_adr_conf:
        ids_adr_conf.append(node.id)
        throughputs_adr_conf.append(node.get_mean_throughput())
    ax_adr_conf.plot(ids_adr_conf, throughputs_adr_conf)

    ax_adr_conf.set_xlabel("Node ID")
    ax_adr_conf.set_ylabel("mean throughput per node")
    ax_adr_conf.set_title("ADR CONF")

    plt.show()

# run_simulation(conf=False, adr=False, training=True, num_nodes=1000)
# run_simulation(conf=True, adr=True, training=False, num_nodes=1000)
# compare_before_and_after(title="energy")
# compare_before_and_after(title="energy per bit")
# compare_before_and_after(title="total energy")
# compare_before_and_after(title="conf + adr")

# compare_before_and_after(title="reward")
# compare_before_and_after(title="energy")
# compare_before_and_after(title="energy per bit")
# compare_before_and_after(title="total energy")
# # compare_before_and_after(title="throughput")

# compare_before_and_after(load=False)
# compare_before_and_after(load=True)

# compare_t_b_and_a()
# compare_thrughput_before_and_after_for_all_nodes()

compare_before_and_after(title="normal")
# compare_before_and_after(title="throughput")
# compare_before_and_after(title="energy")

def paper_1_compare(load=False):

    title = ""
    num_nodes = 1000
    num_days = 10

    locations = list()
    for i in range(num_nodes):
        locations.append(Location(min=0, max=Config.CELL_SIZE, indoor=False))

    configurations = [
        {
            "title": "NO ADR NO CONF",
            "conf": False,
            "adr": False,
            "training": False,
            "num_nodes": num_nodes,
            "reward": title,
            "days": num_days,
            "locations": locations},
        {
            "title": "ADR CONF",
            "conf": True,
            "adr": True,
            "training": False,
            "num_nodes": num_nodes,
            "reward": title,
            "days": num_days,
            "locations": locations},
        {
            "title": "RL",
            "conf": False,
            "adr": False,
            "training": True,
            "num_nodes": num_nodes,
            "reward": title,
            "days": num_days,
            "locations": locations},
    ]

    plt.figure(figsize=(5, 5))

    settings = []
    results = []

    for config_cnt, config in enumerate(configurations):

        nodes, agents, env = init_nodes(conf=config["conf"],
                                        adr=config["adr"],
                                        training=config["training"],
                                        num_nodes=config["num_nodes"],
                                        reward=config["reward"],
                                        locations=config["locations"],
                                        agent_to_nodes=load_agents() if load else None)

        run_nodes(nodes, env, days=config["days"])
        save_agents(agents)

        energy_avg = 0
        num_packets_avg = 0

        for node in nodes:
            energy_avg += node.total_energy_consumed()
            num_packets_avg += node.num_unique_packets_sent

        energy_avg = (energy_avg / num_nodes) / 1000
        num_packets_avg = num_packets_avg / num_nodes
        energy_efficiency = num_packets_avg / energy_avg
        # print(f"Energy Efficiency (Packets/Joule) = {num_packets_avg / energy_avg}")

        settings.append(config["title"])
        results.append(energy_efficiency)

    plt.bar(settings, results, width=0.4)
    plt.ylabel("Energy Efficiency (Packets/Joule)")
    plt.show()

# paper_1_compare(load=False)
# paper_1_compare(load=True)

# fig, axarr = plt.subplots(1,2)
#
# def m1():
#     axarr[0].plot([1,2,3], [4,5,6])
#
# def m2():
#     axarr[1].plot([1,2,3], [4,5,6])
# m1()
# m2()
# plt.show()