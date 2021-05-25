import datetime
import matplotlib.pyplot as plt
import numpy as np
import simpy
import torch
import os

import PropagationModel
from AirInterface2 import AirInterface2
from EnergyProfile import EnergyProfile
from Gateway2 import Gateway2
from Global import Config
from LNS import LNS
from LoRaParameters import LoRaParameters
from Location import Location
from NOMA import NOMA
from Node2 import Node2
from SINRModel import SINRModel
from SNRModel import SNRModel
from agent import LearningAgent
from clustering_nodes import cluster_nodes, search_closest

def plot_time(_env, sim_time):
    while True:
        print('.', end='', flush=True)
        yield _env.timeout(np.round(sim_time / 10))

def location_sorted_insert(array, new_element):

    if (not array or new_element.location.x > array[-1].location.x):
        array.append(new_element)
        return

    if (new_element.location.x < array[0].location.x):
        array.insert(0, new_element)
        return

    length = len(array)
    for i in range(length):
        curr = array[i]
        if (new_element.location.x < curr.location.x):
            elements_with_same_x = i
            for j in range(i, length):
                elements_with_same_x = j
                if (array[j].location.x != new_element.location.x):
                    break
            if (elements_with_same_x != i):
                for k in range(i, elements_with_same_x):
                    curr = array[k]
                    if (new_element.location.y < curr.location.y):
                        array.insert(k, new_element)
                        return
            else:
                array.insert(i, new_element)
                return

def init_nodes(config, agent_to_nodes=None):
    energy_per_bit = 0
    tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}  # measured TX power for each possible TP
    middle = np.round(Config.CELL_SIZE / 2)
    lns_location = Location(x=middle, y=middle, indoor=False)
    env = simpy.Environment()
    lns = LNS(env, lns_location)

    gateways = []
    locations = config["gateway_locations"]
    for gateway_id in range(config["num_gateways"]):
        location = locations[gateway_id]
        gateway = Gateway2(env, location, air_interface=None)
        gateways.append(gateway)
        # location_sorted_insert(gateways, gateway)

    temp = []
    for gateway in gateways:
        temp.append(gateway.location.get_tuple())

    air_interface = AirInterface2(lns, gateways, PropagationModel.LogShadow(), SNRModel(), SINRModel(), env)

    nodes = []

    locations = config["node_locations"]
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

        node = Node2(node_id, energy_profile, lora_param, sleep_time=(8 * payload_size / transmission_rate),
                    process_time=5,
                    location=location,
                    adr=config["adr"],
                    confirmed_messages=config["conf"],
                    training=config["training"],
                    env=env, payload_size=payload_size, air_interface=air_interface,
                    reward_type=config["reward"],
                    state_space=config["state_space"])

        nodes.append(node)
        location_to_node[(location.x, location.y)] = node
        # plt.scatter(location.x, location.y, color='blue')

    # for gateway in gateways:
        # plt.scatter(gateway.location.x, gateway.location.y, color='red')

    # plt.show()

    agents = []

    if (agent_to_nodes == None):

        # axes = plt.gca()
        # axes.set_xlim([0, Config.CELL_SIZE])
        # axes.set_ylim([0, Config.CELL_SIZE])

        # Creation of clusters and assignment of one learning agent per cluster
        print("\nCreating clusters of nodes...")
        clusters = cluster_nodes(nodes, sector_size=config["sector_size"])
        print("Finished creating clusters\n")

        # for cluster in clusters.keys():
        #     for node in clusters[cluster]:
        #         plt.scatter(node.location.x, node.location.y, color='blue')
        #
        # for node in clusters[list(clusters.keys())[13]]:
        #     plt.scatter(node.location.x, node.location.y, color='red')
        #
        # major_ticks = np.arange(0, 1001, config["sector_size"])
        #
        # axes.set_xticks(major_ticks)
        # axes.set_yticks(major_ticks)
        #
        # plt.grid(True)
        # plt.show()

        # start_time = time.perf_counter()
        for (cluster_center_location, cluster) in clusters.items():

            # making sure each agent is assigned to at least one node
            # TODO (problem of uneven distribution of nodes!)
            if len(cluster) > 0:
                agent = LearningAgent(env=env, config=config)
                agent.assign_nodes(cluster)
                agents.append(agent)
    else:
        # No need to check for deep entry of the configuration since
        # only loading and saving of deep models has been implemented so far
        for id in list(agent_to_nodes.keys()):

            agent = LearningAgent(env=env, config=config)
            agent.q_network.load_state_dict(torch.load(f"./model/agent_{id}.pth"))
            agent.q_network.eval()
            agents.append(agent)

            cluster = []
            for location in agent_to_nodes[id]:
                cluster.append(location_to_node[location])

            agent.assign_nodes(cluster)

    return nodes, gateways, agents, env

def run_nodes(nodes, gateways, env, days, noma=True):
    # The simulation is kicked off after agents are assigned to respective clusters
    if noma:
        for node in nodes:
            env.process(node.run_noma())
        for gateway in gateways:
            env.process(gateway.noma.run())
    else:
        for node in nodes:
            env.process(node.run())

    sim_time = 1000*60*60*24*days
    d = datetime.timedelta(milliseconds=sim_time)
    print('Running simulator for {}.'.format(d))
    env.process(plot_time(env, sim_time))
    env.run(until=sim_time)

def health_check(configurations, days):
    print("\n##################          Health check             ###################")
    for config_cnt, config in enumerate(configurations):
        simulation_time = days * 1000 * 60 * 60 * 24
        nodes, gateways, agents, env = init_nodes(config=config)
        run_nodes(nodes, gateways, env, days=days, noma=config["noma"])
    print("\n##################        Health check OKAY          ###################")

def run_configurations(configurations, file_name, save_to_local, main_theme, progression, rest):

    first_run = True
    f_large, axarr_large = None, None
    f_small, axarr_small = None, None
    file_name_large = file_name
    file_name_small = file_name + "_SMALL"

    # plots all configs in configurations on one large plot, and duplicates most important data onto a smaller plot
    for config_cnt, config in enumerate(configurations):
        simulation_time = config["days"]*1000*60*60*24
        nodes, gateways, agents, env = init_nodes(config=config)

        run_nodes(nodes, gateways, env, days=config["days"], noma=config["noma"])

        # basically number of plots in a column (times 2 for all nodes and average)
        num_categories = len(list(nodes[0].rl_measurements.keys()))

        # Initializing all subplots and figures
        if first_run:
            num_columns_large = len(configurations) + 1
            num_rows_large = num_categories * 2 + 3
            # adding 3 additional plots for simulation info at the top, loss and results at the bottom
            f_large, axarr_large = plt.subplots(num_rows_large, num_columns_large, figsize=(num_columns_large * 7, num_rows_large * 5),
                                    sharex='col', sharey='row')

            num_columns_small = num_categories
            num_rows_small = 1
            # adding 3 additional plots for simulation info at the top, loss and results at the bottom
            f_small, axarr_small = plt.subplots(num_rows_small, num_columns_small, figsize=(num_columns_small * 7, num_rows_small * 5),
                                                sharex=True)

        first_run = False

        length_per_parameter = {}
        times_per_parameter = {}
        # sum_per_parameter = {}
        avg_per_parameter = {}

        # Initializing recording for plotting mean value later
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
                              "Sector size — {}\n" \
                .format(
                    config["title"],
                    config["num_nodes"],
                    config["days"],
                    ", ".join(config["state_space"]),
                    agents[0].gamma,
                    agents[0].epsilon,
                    config["sector_size"],
            )

            simulation_results += "Learning rate (Alpha) — {}".format(agents[0].alpha)

            if (len(configurations) == 1):
                axarr_large[0].text(0.1, 0.9, simulation_results, fontsize=20, bbox=props, verticalalignment='top', transform=axarr_large[0].transAxes)
            else:
                axarr_large[0][config_cnt].text(0.1, 0.9, simulation_results, fontsize=20, bbox=props, verticalalignment='top', transform=axarr_large[0][config_cnt].transAxes)
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
                    axarr_large[parameter_cnt * 2 + 1].plot(time, vals)
                    axarr_large[parameter_cnt * 2 + 1].set_xlabel("Time")
                    axarr_large[parameter_cnt * 2 + 1].set_ylabel(f"{parameter} per Node")
                    axarr_large[parameter_cnt * 2 + 1].set_title(config["title"])
                else:
                    axarr_large[parameter_cnt * 2 + 1][config_cnt].plot(time, vals)
                    axarr_large[parameter_cnt * 2 + 1][config_cnt].set_xlabel("Time")
                    axarr_large[parameter_cnt * 2 + 1][config_cnt].set_ylabel(f"{parameter} per Node")
                    axarr_large[parameter_cnt * 2 + 1][config_cnt].set_title(config["title"])

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
                axarr_large[temp].plot(list(avg_per_parameter[parameter].keys()), list(avg_per_parameter[parameter].values()))
                axarr_large[temp].set_xlabel("Time")
                axarr_large[temp].set_ylabel(f"{parameter} AVERAGE per Node")
                axarr_large[temp].set_title(config["title"])
            else:
                axarr_large[temp][config_cnt].plot(list(avg_per_parameter[parameter].keys()),
                                             list(avg_per_parameter[parameter].values()),
                                             label=list(avg_per_parameter[parameter].values())[-1])
                axarr_large[temp][config_cnt].set_xlabel("Time")
                axarr_large[temp][config_cnt].set_ylabel(f"{parameter} AVERAGE per Node")
                axarr_large[temp][config_cnt].set_title(config["title"])
                axarr_large[temp][config_cnt].legend()

                # Plotting all mean plots together in the last solumn of the large plot
                axarr_large[temp][-1].plot(list(avg_per_parameter[parameter].keys()),
                                             list(avg_per_parameter[parameter].values()), label=config["label"])
                axarr_large[temp][-1].set_xlabel("Time")
                axarr_large[temp][-1].set_ylabel(f"{parameter} AVERAGE per Node")
                axarr_large[temp][-1].set_title("Comparison")
                axarr_large[temp][-1].legend()


                # Plotting all mean plots together in the small plot

                axarr_small[parameter_cnt].plot(list(avg_per_parameter[parameter].keys()),
                                        list(avg_per_parameter[parameter].values()), label=config["label"])
                axarr_small[parameter_cnt].set_xlabel("Time")
                axarr_small[parameter_cnt].set_ylabel(f"{parameter} - mean per node")
                axarr_small[parameter_cnt].set_title("")
                axarr_small[parameter_cnt].legend()

        # Plotting losses for deep learning agents
        agent_id = 0
        for agent in agents:
            losses = agent.losses

            if (len(configurations) == 1):
                axarr_large[-2].plot(list(losses.keys()), list(losses.values()), label=agent_id)
                agent_id += 1
                axarr_large[-2].set_xlabel("Time")
                axarr_large[-2].set_ylabel("Loss per Agent")
                axarr_large[-2].set_title(config["title"])
            else:
                axarr_large[-2][config_cnt].plot(list(losses.keys()), list(losses.values()), label=agent_id)
                agent_id += 1
                axarr_large[-2][config_cnt].set_xlabel("Time")
                axarr_large[-2][config_cnt].set_ylabel("Loss per Agent")
                axarr_large[-2][config_cnt].set_title(config["title"])

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
                axarr_large[-1].text(0.1, 0.9, simulation_results, fontsize=20, bbox=props, verticalalignment='top',
                              transform=axarr_large[-1].transAxes)
            else:
                axarr_large[-1][config_cnt].text(0.1, 0.9, simulation_results, fontsize=20, bbox=props, verticalalignment='top',
                                          transform=axarr_large[-1][config_cnt].transAxes)
        simulation_results()

    def save_figures():

        if (save_to_local):
            # assuming that the local machine is running Windows
            now = datetime.datetime.now()
            current_date = now.strftime("%d.%m.%Y")
            current_hour = now.strftime("%H")
            save_dir_date = f"../Plot Diary/{current_date}/"
            save_dir = f"../Plot Diary/{current_date}/{current_hour}/"
            if (not os.path.exists(save_dir_date)):
                os.mkdir(save_dir_date)
        else:
            # assuming that the NOT local machine is running Linux
            save_dir = "./plots/"

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        name = save_dir + file_name_large + ".png"
        f_large.savefig(name)
        print(f"\nSaved plot {name}")

        name = save_dir + file_name_small + ".png"
        f_small.savefig(name)
        print(f"Saved plot {name}")

        name = save_dir + file_name_large + ".txt"
        with open(name, 'w') as f:

            def dict_to_str(dict, depth=0):
                ret = "{"
                for key, val in dict.items():
                    tabs = "\t" + (depth * "\t")
                    ret += f"\n{tabs}{key}: {val};"
                tabs = (depth * "\t")
                return ret + f"\n{tabs}" + "}"

            def progression_to_str(progression):
                ret = "["
                for p in progression:
                    ret += f"\n\t{dict_to_str(p, depth=1)},"
                return ret + "\n]"

            main_theme_dict = dict_to_str(main_theme)
            progression_dict = progression_to_str(progression)
            rest_dict = dict_to_str(rest)

            input  = f"MAIN THEME \n\n {main_theme_dict} \n\n PROGRESSION \n\n {progression_dict} \n\n THE REST \n\n {rest_dict}"
            f.write(input)
            f.close()
            print(f"Saved file {name}\n")
    save_figures()

    plt.show()

# Standard configuration values
num_nodes = 1000
node_locations = list()
for i in range(num_nodes):
    node_locations.append(Location(min=0, max=Config.CELL_SIZE, indoor=False))

num_gateways = 5
gateway_locations = list()

size = Config.CELL_SIZE
half = size / 2
gen = np.random

x, y = gen.rand() * half, gen.rand() * half
gateway_locations.append(Location(x=x, y=y, indoor=False))

x, y = gen.rand() * half, half + gen.rand() * half
gateway_locations.append(Location(x=x, y=y, indoor=False))

x, y = half + gen.rand() * half, gen.rand() * half
gateway_locations.append(Location(x=x, y=y, indoor=False))

x, y = half + gen.rand() * half, half + gen.rand() * half
gateway_locations.append(Location(x=x, y=y, indoor=False))

x, y = gen.rand() * size, gen.rand() * size
gateway_locations.append(Location(x=x, y=y, indoor=False))

# gateway_locations.append(Location(x=1, y=1, indoor=False))
# gateway_locations.append(Location(x=3, y=2, indoor=False))
# gateway_locations.append(Location(x=4, y=1, indoor=False))
# gateway_locations.append(Location(x=5, y=1, indoor=False))
# gateway_locations.append(Location(x=4, y=4, indoor=False))
# gateway_locations.append(Location(x=4, y=12, indoor=False))
# gateway_locations.append(Location(x=2, y=5, indoor=False))

normal_reward = "normal"
thoughput_reward = "throughput"
energy_reward = "energy"

def generate_config(config):
    standard_body = {
        "title": "",
        "file_name": "",
        "label": "",
        "conf": False,
        "adr": False,
        "training": False,
        "deep": False,
        "depth": 2,
        "sarsa": False,
        "mc": False,
        "replay_buffer": False,
        "double_deep": False,
        "load": False,
        "save": False,
        "num_nodes": num_nodes,
        "num_gateways": num_gateways,
        "reward": normal_reward,
        "days": 10,
        "node_locations": node_locations,
        "gateway_locations": gateway_locations,
        "state_space": ["tp", "sf", "channel"],
        "sector_size": 100,
        "gamma": 0.5,
        "slow_sf": False,
        "slow_tp": False,
        "slow_channel": False,
        "noma": True,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "GLIE": False,
        "Robbins-Monroe": False,
        "epsilon_decay_rate": -1,
        "alpha_decay_rate": -1,
        "expected_sarsa": False,
    }

    for (key, val) in config.items():
        standard_body[key] = val

    return standard_body

no_adr_no_conf_config = generate_config({
    "title": "NO ADR NO CONF",
    "label": "nothing",
})

adr_conf_config = generate_config({
    "title": "ADR CONF",
    "conf": True,
    "adr": True,
    "label": "adr, conf",
})
# plot_air_packages(configurations=configurations)

# config_global = [
#     # "Replay buffer and double deep",
#     [
#         "Applying all optimizations with sector size 50",
#         # no_adr_no_conf_config,
#         {
#             "title": "Deep Q learning",
#             "training": True,
#             "deep": True,
#             "double_deep": True,
#             "replay_buffer": True,
#             "GLIE": True,
#             "Robbins-Monroe": True,
#             "state_space": ["tp", "sf", "channel", "sinr", "rss", "energy"],
#         },
#     ],
# ]

# Hypertuning progression
progression = [
    {
        "title": "",
        "label": "default",
    },
    {
        "title": "",
        "label": "decay",
        "GLIE": True,
        "epsilon_decay_rate": 1,
        "Robbins-Monroe": True,
        "alpha_decay_rate": 1,
    }]

config_global = [
    [
        "deep_sarsa_decay",
        {
            "title": "Deep Q-learning",
            "training": True,
            "deep": True,
        },
        progression
    ],
]

# Still to try buffer with new optimizations
for i in range(len(config_global)):

    simulation_name, main_theme, progression = config_global[i]
    configs_combination_per_simulation = []

    for curr_config in progression:

        temp = {}

        # copying from curr_config. we still want to keep progression intact for printing purposes
        for (key, val) in curr_config.items():
            temp[key] = val

        # Applying the main theme, combining 2 titles together
        title = main_theme["title"] + ", " + temp["title"]
        for (key, val) in main_theme.items():
            temp[key] = val
        temp["title"] = title

        temp = generate_config(temp)
        configs_combination_per_simulation.append(temp)

    print(f"\n\n\n STARTING SIMULATION: \t\t\t {simulation_name} \n\n\n")
    # try:
    health_check(configurations=configs_combination_per_simulation, days=0.1)
    run_configurations(configurations=configs_combination_per_simulation,
                       save_to_local=None,
                       file_name=config_global[i][0],
                       main_theme=main_theme,
                       progression=progression,
                       rest=configs_combination_per_simulation[0],
                       )
    # except Exception as e:
    #     print(f"THERE WAS A PROBLEM RUNNING SIMULATION — {config_global[i][0]}")
    #     print(f"THERE WAS A PROBLEM RUNNING SIMULATION — {config_global[i][0]}")
    #     print(f"THERE WAS A PROBLEM RUNNING SIMULATION — {config_global[i][0]}")
    #     print(e)
    #
    # print(
    #     "\n\n\n#################################################################################################\n\n\n")
