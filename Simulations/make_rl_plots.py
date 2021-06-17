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
from agent import LearningAgent
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

def plot_time(_env):
    while True:
        print('.', end='', flush=True)
        yield _env.timeout(np.round(simulation_time / 10))


tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}  # measured TX power for each possible TP
middle = np.round(Config.CELL_SIZE / 2)
gateway_location = Location(x=middle, y=middle, indoor=False)

num_nodes = 1000
num_of_simulations = 5

cluster_sizes = range(50, 200, 25)

simultation_results = dict()
gateway_results = dict()
air_interface_results = dict()

# for payload_size in payload_sizes:
#     simultation_results[payload_size] = pd.DataFrame()
#     gateway_results[payload_size] = pd.DataFrame()
#     air_interface_results[payload_size] = pd.DataFrame()


mu_energy = dict()
sigma_energy = dict()

# for payload_size in payload_sizes:
#     mu_energy[payload_size] = dict()
#     sigma_energy[payload_size] = dict()
#     for path_loss_variance in path_loss_variances:
#         mu_energy[payload_size][path_loss_variance] = 0
#         sigma_energy[payload_size][path_loss_variance] = 0

for s in cluster_sizes:
    simultation_results[s] = pd.DataFrame()
    gateway_results[s] = pd.DataFrame()
    air_interface_results[s] = pd.DataFrame()
    mu_energy[s] = 0
    sigma_energy[s] = 0

# load locations:
with open('locations.pkl', 'rb') as filehandler:
    locations_per_simulation = pickle.load(filehandler)

def make_pickles(adr, confirmed_messages, rl):

    # default values
    payload_size = 25
    path_loss_variance = 7.8
    rl_plots = None

    for n_sim in range(num_of_simulations):
        print("Staring simulation: {}".format(n_sim))

        locations = locations_per_simulation[n_sim]

        for cluster_size in cluster_sizes:

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
                            adr=adr,
                            confirmed_messages=confirmed_messages,
                            training=rl,
                            base_station=gateway, rl_plots=rl_plots, env=env, payload_size=payload_size,
                            air_interface=air_interface)

                nodes.append(node)

            print("Creating clusters of nodes...")
            clusters = cluster_nodes(nodes, sector_size=50)
            print("Finished creating clusters")

            for (cluster_center_location, cluster) in clusters.items():
                agent = DeepLearningAgent(depth=4, gamma=0.9, lr=0.001, epsilon=0.9)

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
                    num_nodes * num_of_simulations)
            data_gateway = gateway.get_simulation_data(name=cluster_size) / (num_nodes * num_of_simulations)
            data_air_interface = air_interface.get_simulation_data(name=cluster_size) / (
                    num_nodes * num_of_simulations)

            if not cluster_size in simultation_results[cluster_size].index:
                simultation_results[cluster_size] = simultation_results[cluster_size].append(data_node)
                gateway_results[cluster_size] = gateway_results[cluster_size].append(data_gateway)
                air_interface_results[cluster_size] = air_interface_results[cluster_size].append(data_air_interface)

            else:
                simultation_results[cluster_size].loc[[cluster_size]] += data_node
                gateway_results[cluster_size].loc[[cluster_size]] += data_gateway
                air_interface_results[cluster_size].loc[[cluster_size]] += data_air_interface

            # tmp = simultation_results[payload_size].index

            mu, sigma = Node.get_energy_per_byte_stats(nodes, gateway)
            print("mu: {}, sigma: {}".format(mu, sigma))
            mu_energy[cluster_size] += mu / num_of_simulations
            sigma_energy[cluster_size] += sigma / num_of_simulations

            # END loop path_loss_variances

            # Printing experiment parameters
            print('{} payload size '.format(payload_size))
            print('{} transmission rate'.format(transmission_rate))
            print('{} ADR'.format(adr))
            print('{} confirmed msgs'.format(confirmed_messages))
            print('{}m cell size'.format(cell_size))

            # END loop payload_sizes

    # END LOOP SIMULATION
    adr_prefix = "adr"
    if (not adr):
        adr_prefix = "no_adr"

    conf_prefix = "conf"
    if (not confirmed_messages):
        conf_prefix = "no_conf"

    rl_prefix = "rl"
    if (not rl):
        conf_prefix = "no_rl"

    directory = f'./Measurements/reinforcement_learning/{adr_prefix}_{conf_prefix}_{rl_prefix}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for cluster_size in cluster_sizes:
        print(cluster_size)
        simultation_results[cluster_size]['mean_energy_per_byte'] = (mu_energy[cluster_size])
        simultation_results[cluster_size]['sigma_energy_per_byte'] = (sigma_energy[cluster_size])
        simultation_results[cluster_size]['UniqueBytes'] = simultation_results[cluster_size].UniquePackets * payload_size
        simultation_results[cluster_size]['CollidedBytes'] = simultation_results[
                                                                 cluster_size].CollidedPackets * payload_size
        simultation_results[cluster_size]['RewardScore'] = simultation_results[cluster_size].RewardScore

        simultation_results[cluster_size].to_pickle(
            directory + 'simulation_results_node_{}'.format(cluster_size))
        print(simultation_results[cluster_size])

        gateway_results[cluster_size].to_pickle(
            directory + 'gateway_results_{}'.format(cluster_size))
        print(gateway_results[cluster_size])

        air_interface_results[cluster_size].to_pickle(
            directory + 'air_interface_results_{}'.format(cluster_size))
        print(air_interface_results[cluster_size])

# make_pickles(adr=True, confirmed_messages=False, rl=False)
# print("-----------MAKE 1 IS DONE----------")
# make_pickles(adr=False, confirmed_messages=True, rl=False)
# print("-----------MAKE 2 IS DONE----------")
make_pickles(adr=False, confirmed_messages=False, rl=True)
print("-----------MAKE 3 IS DONE----------")

