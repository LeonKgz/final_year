import datetime

import matplotlib.pyplot as plt
import numpy as np
import simpy
import torch
import subprocess

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

def init_nodes(config, agent_to_nodes=None):
    energy_per_bit = 0
    tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}  # measured TX power for each possible TP
    middle = np.round(Config.CELL_SIZE / 2)
    gateway_location = Location(x=middle, y=middle, indoor=False)
    # plt.scatter(middle, middle, color='red')
    env = simpy.Environment()
    gateway = Gateway(env, gateway_location)
    nodes = []
    air_interface = AirInterface(gateway, PropagationModel.LogShadow(), SNRModel(), SINRModel(), env)

    locations = config["locations"]
    if (agent_to_nodes != None):
        locations = []
        for locs in list(agent_to_nodes.values()):
            locations += [Location(x=x, y=y, indoor=False) for (x, y) in locs]

    location_to_node = {}

    for node_id in range(config["num_nodes"]):
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
                    adr=config["adr"],
                    confirmed_messages=config["conf"],
                    training=config["training"],
                    base_station=gateway, env=env, payload_size=payload_size, air_interface=air_interface,
                    reward_type=config["reward"],
                    state_space=config["state_space"])

        nodes.append(node)
        location_to_node[(location.x, location.y)] = node

    agents = []

    if (agent_to_nodes == None):
        # Creation of clusters and assignment of one learning agent per cluster
        print("Creating clusters of nodes...")
        clusters = cluster_nodes(nodes, sector_size=config["sector_size"])
        print("Finished creating clusters")

        for (cluster_center_location, cluster) in clusters.items():
            agent = DeepLearningAgent(env=env, depth=4, state_space_dimensions=len(config["state_space"]), gamma=0.5,
                                      epsilon=0.5, lr=0.001, sarsa=config["sarsa"],
                                      replay_buffer=config["replay_buffer"],
                                      double_deep=config["double_deep"]) \
                if config["deep"] else \
                    LearningAgent(env=env, gamma=0.5, epsilon=0.5, alpha=0.5, sarsa=config["sarsa"], mc=config["mc"])

            # making sure each agent is assigned to at least one node
            # TODO (problem of uneven distribution of nodes!)
            # if len(cluster) > 0:
            agent.assign_nodes(cluster)
            agents.append(agent)
    else:
        # No need to check for deep entry of the configuration since
        # only loading and saving of deep models has been implemented so far
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

def run_nodes(nodes, env, days):
    # The simulation is kicked off after agents are assigned to respective clusters
    for node in nodes:
        env.process(node.run())
    sim_time = 1000*60*60*24*days
    d = datetime.timedelta(milliseconds=sim_time)
    print('Running simulator for {}.'.format(d))
    env.process(plot_time(env, sim_time))
    env.run(until=sim_time)

def plot_air_packages(configurations):

    first_run = True
    f, axarr = None, None

    for config_cnt, config in enumerate(configurations):
        simulation_time = config["days"]*1000*60*60*24
        nodes, agents, env = init_nodes(config=config)
        run_nodes(nodes, env, days=config["days"])

def compare_before_and_after(configurations, save_to_local=False):

    first_run = True
    f, axarr = None, None

    for config_cnt, config in enumerate(configurations):
        simulation_time = config["days"]*1000*60*60*24
        nodes, agents, env = init_nodes(config=config)

        run_nodes(nodes, env, days=config["days"])

        # basically number of plots in a column (times 2 for all nodes and average)
        num_categories = len(list(nodes[0].rl_measurements.keys()))

        if first_run:
            num_columns = len(configurations)
            num_rows = num_categories * 2 + 3
            # adding 3 additional plots for simulation info at the top, loss and results at the bottom
            f, axarr = plt.subplots(num_rows, num_columns, figsize=(num_columns * 7, num_rows * 5),
                                    sharex='col', sharey='row')

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

        def simulation_info():
            # Display information about current configuration (at the top of all the plots)
            props = dict(alpha=0.6)

            simulation_results = "Flavour — {}\n" \
                              "Number of nodes — {}\n" \
                              "Simulation time — {} days\n" \
                              "State space — [ {} ]\n" \
                              "Gamma — {}\n" \
                              "Epsilon — {}\n" \
                .format(
                    config["title"],
                    config["num_nodes"],
                    config["days"],
                    ", ".join(config["state_space"]),
                    agents[0].gamma,
                    agents[0].epsilon
                )

            if (config["deep"]):
                simulation_results += "Learning rate — {}".format(agents[0].lr)
            else:
                simulation_results += "Alpha — {}".format(agents[0].alpha)

            if (len(configurations) == 1):
                axarr[0].text(0.1, 0.9, simulation_results, fontsize=20, bbox=props, verticalalignment='top', transform=axarr[0].transAxes)
            else:
                axarr[0][config_cnt].text(0.1, 0.9, simulation_results, fontsize=20, bbox=props, verticalalignment='top', transform=axarr[0][config_cnt].transAxes)
        simulation_info()

        # First simply plotting all the nodes at the same time, while also recording data for averaging later
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

        energy_avg = (energy_avg / config["num_nodes"]) / 1000
        num_packets_avg = num_packets_avg / config["num_nodes"]

        # Now plotting the mean values for above
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

            temp = (parameter_cnt * 2) + 2

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

        # Plotting losses for deep learning agents
        agent_id = 0
        for agent in agents:
            losses = agent.losses

            if (len(configurations) == 1):
                axarr[-2].plot(list(losses.keys()), list(losses.values()), label=agent_id)
                agent_id += 1
                axarr[-2].set_xlabel("Time")
                axarr[-2].set_ylabel("Loss per Agent")
                axarr[-2].set_title(config["title"])
            else:
                axarr[-2][config_cnt].plot(list(losses.keys()), list(losses.values()), label=agent_id)
                agent_id += 1
                axarr[-2][config_cnt].set_xlabel("Time")
                axarr[-2][config_cnt].set_ylabel("Loss per Agent")
                axarr[-2][config_cnt].set_title(config["title"])

        def simulation_results():

            temp = []
            for node in nodes:
                throughputs = node.rl_measurements["throughputs"]
                temp.append(list(throughputs.values())[-1])
                # mean_adr_conf = np.mean(list(rewards.values()))
            mean_throughput = np.mean(temp)

            # print(f"\n Mean throughput (at the end of the simulation) for " + str(config["title"]) + " is " + str(mean))
            # print(f"\n Energy Efficiency (Packets/Joule) = {}")

            # Display information about simulation results (at the bottom)
            props = dict(alpha=0.6)

            simulation_results = "Throughput — {:.2f}\n" \
                              "EE (Packets/Joule) — {:.2f}\n" \
                .format(
                mean_throughput,
                num_packets_avg / energy_avg,
            )

            if (len(configurations) == 1):
                axarr[-1].text(0.1, 0.9, simulation_results, fontsize=20, bbox=props, verticalalignment='top',
                              transform=axarr[-1].transAxes)
            else:
                axarr[-1][config_cnt].text(0.1, 0.9, simulation_results, fontsize=20, bbox=props, verticalalignment='top',
                                          transform=axarr[-1][config_cnt].transAxes)
        simulation_results()

    # if (save_to_local):
    #    subprocess.run()


    plt.show()



# Standard configuration values
num_nodes = 1000
locations = list()
for i in range(num_nodes):
    locations.append(Location(min=0, max=Config.CELL_SIZE, indoor=False))

normal_reward = "normal"
thoughput_reward = "throughput"
energy_reward = "energy"

def generate_config(config):
    standard_body = {
      "title": "",
      "conf": False,
      "adr": False,
      "training": False,
      "deep": False,
      "sarsa": False,
      "mc": False,
      "replay_buffer": False,
      "double_deep": False,
      "load": False,
      "save": False,
      "num_nodes": num_nodes,
      "reward": normal_reward,
      "days": 10,
      "locations": locations,
      "state_space": [],
      "sector_size": 100
    }

    for (key, val) in config.items():
        standard_body[key] = val

    return standard_body

no_adr_no_conf_config = generate_config({
    "title": "NO ADR NO CONF"})

adr_conf_config = generate_config({
    "title": "ADR CONF",
    "conf": True,
    "adr": True})

configurations = [
    no_adr_no_conf_config,
    adr_conf_config,
    generate_config({
        "title": "Q learning",
        "training": True,
        "deep": True,
        "state_space": ["tp", "sf", "channel"]}),
    generate_config({
        "title": "Q-learning + Replay + Target",
        "training": True,
        "deep": True,
        "replay_buffer": True,
        "double_deep": True,
        "state_space": ["tp", "sf", "channel"]})
]

compare_before_and_after(configurations=configurations, save_to_local=True)
# plot_air_packages(configurations=configurations)
