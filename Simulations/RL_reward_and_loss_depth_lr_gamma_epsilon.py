import datetime

import matplotlib.pyplot as plt
import numpy as np
import simpy

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
from agent import DeepLearningAgent
from clustering_nodes import cluster_nodes


def plot_time(_env):
    while True:
        print('.', end='', flush=True)
        yield _env.timeout(np.round(Config.SIMULATION_TIME / 10))

def make_plots(depth, lr, gamma, epsilon, cluster_size=75, num_nodes=1000, sim_time=1000*60*60*24*10):
    energy_per_bit = 0
    payload_size = 25
    tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}  # measured TX power for each possible TP
    middle = np.round(Config.CELL_SIZE / 2)
    gateway_location = Location(x=middle, y=middle, indoor=False)
    env = simpy.Environment()
    gateway = Gateway(env, gateway_location)
    nodes = []
    air_interface = AirInterface(gateway, PropagationModel.LogShadow(), SNRModel(), SINRModel(), env)
    rl_plots = RL_plots(env=env)

    for node_id in range(num_nodes):
        location = Location(min=0, max=Config.CELL_SIZE, indoor=False)
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
                    adr=False,
                    confirmed_messages=False,
                    training=True,
                    base_station=gateway, env=env, payload_size=payload_size, air_interface=air_interface)

        nodes.append(node)

    # Creation of clusters and assignment of one learning agent per cluster
    print("Creating clusters of nodes...")
    clusters = cluster_nodes(nodes, sector_size=cluster_size)
    print("Finished creating clusters")

    agents = []
    for (cluster_center_location, cluster) in clusters.items():
        agent = DeepLearningAgent(env=env, depth=depth, lr=lr, gamma=gamma, epsilon=epsilon)
        # making sure each agent is assigned to at least one node
        # TODO get rid of empty clusters earlier (problem of uneven distribution of nodes!)
        if len(cluster) > 0:
            agent.assign_nodes(cluster)
            agents.append(agent)

    # The simulation is kicked off after agents are assigned to respective clusters
    for node in nodes:
        env.process(node.run())

    env.process(plot_time(env))

    d = datetime.timedelta(milliseconds=sim_time)
    print('Running simulator for {}.'.format(d))
    env.run(until=sim_time)

    rl_plots.plot_rewards_and_losses(nodes=nodes, agents=agents,
                                     depth=depth, lr=lr, epsilon=epsilon, gamma=gamma,
                                     show=False, save=True)
    print(f'Simulator is done for — depth={depth}_lr={lr}_epsilon={epsilon}_gamma={gamma}_.png')

# depth_range = range(3, 5)
depth = 3
lr_range = [0.001, 0.01, 0.1]
epsilon_range = [0.1, 0.5, 0.9]
gamma_range = [0.1, 0.5, 0.9]


lr_range = [0.1]
epsilon_range = [0.1, 0.5, 0.9]
gamma_range = [0.1, 0.5, 0.9]

# for depth in depth_range:
for lr in lr_range:
    for epsilon in epsilon_range:
        for gamma in gamma_range:
            make_plots(depth=depth, lr=lr, gamma=gamma, epsilon=epsilon)
