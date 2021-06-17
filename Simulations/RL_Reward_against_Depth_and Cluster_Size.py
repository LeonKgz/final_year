import datetime

import numpy as np
import pandas as pd
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
import os
import pickle

# The console attempts to auto-detect the width of the display area, but when that fails it defaults to 80
# characters. This behavior can be overridden with:
desired_width = 320
pd.set_option('display.width', desired_width)

transmission_rate = 0.02e-3  # 12*8 bits per hour (1 typical packet per hour)
simulation_time = 10 * 24 * 60 * 60 * 1000
cell_size = 1000
num_nodes = 1000
payload_size = 25
path_loss_variance = 7.8 # sigma

num_simulations = 10
model_depth_range = range(2, 10)
learning_rate_range = []
epsilon_range = []
gamma_range = []
cluster_size_range = range(50, 200, 25)

locations_per_simulation = list()
for num_sim in range(num_simulations):
    locations = list()
    for i in range(num_nodes):
        locations.append(Location(min=0, max=cell_size, indoor=False))
    locations_per_simulation.append(locations)

def plot_time(_env):
    while True:
        print('.', end='', flush=True)
        yield _env.timeout(np.round(simulation_time / 10))

tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}  # measured TX power for each possible TP
middle = np.round(Config.CELL_SIZE / 2)
gateway_location = Location(x=middle, y=middle, indoor=False)

simulation_results = {}
gateway_results = {}
air_interface_results = {}
mu_energy = {}
sigma_energy = {}

for depth in model_depth_range:
    simulation_results[depth] = pd.DataFrame()
    gateway_results[depth] = pd.DataFrame()
    air_interface_results[depth] = pd.DataFrame()

for depth in model_depth_range:
    mu_energy[depth] = {}
    sigma_energy[depth] = {}
    for cluster_size in cluster_size_range:
        mu_energy[depth][cluster_size] = 0
        sigma_energy[depth][cluster_size] = 0

for n_sim in range(num_simulations):
    print(f"Staring simulation: {n_sim}")
    locations = locations_per_simulation[n_sim]

    for depth in model_depth_range:

        for cluster_size in cluster_size_range:

            env = simpy.Environment()
            gateway = Gateway(env, gateway_location, max_snr_adr=False, avg_snr_adr=True)
            nodes = []
            air_interface = AirInterface(gateway, PropagationModel.LogShadow(std=path_loss_variance), SNRModel(), SINRModel(), env)
            np.random.shuffle(locations)

            for node_id in range(num_nodes):

                energy_profile = EnergyProfile(5.7e-3, 15, tx_power_mW,
                                               rx_power={'pre_mW': 8.25, 'pre_ms': 3.4, 'rx_lna_on_mW': 36.96,
                                                         'rx_lna_off_mW': 34.65,
                                                         'post_mW': 8.3, 'post_ms': 10.7})
                lora_param = LoRaParameters(freq=np.random.choice(LoRaParameters.DEFAULT_CHANNELS),
                                            sf=np.random.choice(LoRaParameters.SPREADING_FACTORS),
                                            bw=125, cr=5, crc_enabled=1, de_enabled=0, header_implicit_mode=0, tp=14)

                node = Node(node_id, energy_profile, lora_param, sleep_time=(8 * payload_size / transmission_rate),
                            process_time=5,
                            location=locations[node_id],
                            adr=False,
                            confirmed_messages=False,
                            training=True,
                            base_station=gateway, rl_plots=None, env=env, payload_size=payload_size,
                            air_interface=air_interface)
                nodes.append(node)

            print("Creating clusters of nodes...")
            clusters = cluster_nodes(nodes, sector_size=200)
            print("Finished creating clusters")

            for (cluster_center_location, cluster) in clusters.items():
                agent = DeepLearningAgent(depth=depth)

                # making sure each agent is assigned to at least one node
                # TODO get rid of empty clusters earlier (problem of uneven distribution of nodes!)
                if len(cluster) > 0:
                    agent.assign_nodes(cluster)

            # The simulation is kicked off after agents are assigned to respective clusters
            for node in nodes:
                env.process(node.run())

            # END adding nodes to simulation

            env.process(plot_time(env))

            d = datetime.timedelta(milliseconds=simulation_time)
            print('Running simulator for {}.'.format(d))
            env.run(until=simulation_time)
            print('Simulator is done for cluster size {}'.format(cluster_size))

            data_node = Node.get_mean_simulation_data_frame(nodes, name=cluster_size) / (
                    num_nodes * num_simulations)
            data_gateway = gateway.get_simulation_data(name=cluster_size) / (num_nodes * num_simulations)
            data_air_interface = air_interface.get_simulation_data(name=cluster_size) / (
                    num_nodes * num_simulations)

            if not cluster_size in simulation_results[depth].index:
                simulation_results[depth] = simulation_results[depth].append(data_node)
                gateway_results[depth] = gateway_results[depth].append(data_gateway)
                air_interface_results[depth] = air_interface_results[depth].append(data_air_interface)

            else:
                simulation_results[depth].loc[[cluster_size]] += data_node
                gateway_results[depth].loc[[cluster_size]] += data_gateway
                air_interface_results[depth].loc[[cluster_size]] += data_air_interface

            mu, sigma = Node.get_energy_per_byte_stats(nodes, gateway)
            print("mu: {}, sigma: {}".format(mu, sigma))
            mu_energy[depth][cluster_size] += mu / num_simulations
            sigma_energy[depth][cluster_size] += sigma / num_simulations

storage_dir = f'./Measurements/reinforcement_learning_v2/'
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

for depth in model_depth_range:
    print(depth)
    simulation_results[depth]['mean_energy_per_byte'] = list(mu_energy[depth].values())
    simulation_results[depth]['sigma_energy_per_byte'] = list(sigma_energy[depth].values())
    simulation_results[depth]['UniqueBytes'] = simulation_results[depth].UniquePackets * payload_size
    simulation_results[depth]['CollidedBytes'] = simulation_results[depth].CollidedPackets * payload_size
    simulation_results[depth]['RewardScore'] = simulation_results[depth].RewardScore

    simulation_results[depth].to_pickle(
        storage_dir + 'simulation_results_node_{}'.format(depth))
    print(simulation_results[depth])

    gateway_results[depth].to_pickle(
        storage_dir + 'gateway_results_{}'.format(depth))
    print(gateway_results[depth])

    air_interface_results[depth].to_pickle(
        storage_dir + 'air_interface_results_{}'.format(depth))
    print(air_interface_results[depth])


