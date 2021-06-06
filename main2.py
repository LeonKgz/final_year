import datetime
import matplotlib.pyplot as plt
import numpy as np
import simpy
import torch
import os
import seaborn as sns
import glob
import pickle

import PropagationModel
from AirInterface import AirInterface
from EnergyProfile import EnergyProfile
from Gateway import Gateway
from Global import Config
from LoRaParameters import LoRaParameters
from Location import Location
from NOMA import NOMA
from Node import Node
from SINRModel import SINRModel
from SNRModel import SNRModel
from agent import LearningAgent
from clustering_nodes import cluster_nodes, search_closest

def plot_time(_env, sim_time):
    while True:
        print('.', end='', flush=True)
        yield _env.timeout(np.round(sim_time / 10))


def save_agents(agents):

    files = glob.glob('./models/*')
    for f in files:
        os.remove(f)

    agent_to_node = {}

    id = 0
    for agent in agents:
        torch.save(agent.q_network.state_dict(), f"./models/agent_{id}.pth")
        agent_to_node[id] = [(node.location.x, node.location.y) for node in agent.nodes]
        id += 1

    with open('./models/agent_to_node', 'wb') as file:
        pickle.dump(agent_to_node, file, protocol=pickle.HIGHEST_PROTOCOL)

    # pd_dict = pd.DataFrame.from_dict(agent_to_node)
    # pd_dict.to_pickle("./model/agent_to_node")

# loads a dicitonary mapping agent model ids to locations of nodes (in a cluster)
# associated with that model (learning agent)
def load_agents(clusters="./models/agent_to_node"):

    with open('./models/agent_to_node', 'rb') as file:
        ret = pickle.load(file)
        return ret

    # return pd.read_pickle(clusters)

def init_nodes(config, agent_to_nodes=None):
    energy_per_bit = 0
    tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}  # measured TX power for each possible TP
    middle = np.round(Config.CELL_SIZE / 2)
    gateway_location = Location(x=middle, y=middle, indoor=False)
    # plt.scatter(middle, middle, color='red')
    env = simpy.Environment()
    gateway = Gateway(env, gateway_location, config=config)
    nodes = []
    air_interface = AirInterface(gateway, PropagationModel.LogShadow(std=config["path_variance"]), SNRModel(), SINRModel(), env, config=config)
    noma = NOMA(gateway=gateway, air_interface=air_interface, env=env, config=config)

    if (config["static_locations"]):
        locations = config["locations"]
    else:
        locations = []
        for i in range(config["num_nodes"]):
            locations.append(Location(min=0, max=Config.CELL_SIZE, indoor=False))

    if (agent_to_nodes != None):
        locations = []
        for locs in list(agent_to_nodes.values()):
            locations += [Location(x=x, y=y, indoor=False) for (x, y) in locs]

    location_to_node = {}

    for node_id in range(config["num_nodes"]):
        # location = Location(min=0, max=Config.CELL_SIZE, indoor=False)
        location = locations[node_id]
        payload_size = config["payload_size"]
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
                    base_station=gateway, noma=noma, env=env, payload_size=payload_size, air_interface=air_interface,
                    reward_type=config["reward"],
                    state_space=config["state_space"],
                    config=config)

        nodes.append(node)
        location_to_node[(location.x, location.y)] = node

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
            agent.q_network.load_state_dict(torch.load(f"./models/agent_{id}.pth"))
            agent.q_network.eval()
            agents.append(agent)

            cluster = []
            for location in agent_to_nodes[id]:
                cluster.append(location_to_node[location])

            agent.assign_nodes(cluster)

    return nodes, agents, env

def run_nodes(nodes, env, days, noma=True):
    # The simulation is kicked off after agents are assigned to respective clusters
    if noma:
        for node in nodes:
            env.process(node.run_noma())
        env.process(nodes[0].noma.run())
    else:
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

def health_check(configurations, days):
    print("\n##################          Health check             ###################")
    for config_cnt, config in enumerate(configurations):
        simulation_time = days * 1000 * 60 * 60 * 24
        nodes, agents, env = init_nodes(config=config)
        run_nodes(nodes, env, days=days, noma=config["noma"])
    print("\n##################        Health check OKAY          ###################")

def energy_plots(configurations, file_name, save_to_local):

    # Borrowed code START
    sns.axes_style('white')
    color = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    for i in range(len(color)):
        r, g, b = color[i]
        color[i] = (r / 255., g / 255., b / 255.)

    payload_sizes = range(5, 55, 5)
    path_loss_variances = [0, 5, 7.8, 15, 20]

    payload_sizes = [25, 30, 35]
    path_loss_variances = [7.8, 20]

    i = 0
    j = 0
    colors = dict()
    for var in path_loss_variances:
        colors[var] = color[j]
        j += 1

    # Borrowed code END

    name = file_name + "_energy_plots"

    num_plots = len(configurations)

    f, axarr = plt.subplots(1, num_plots, sharex=True, sharey=False, figsize=(4*num_plots, 4))

    results = {}
    for v in path_loss_variances:
        results[v] = []
        # for p in payload_sizes:
        #     results[v][p] = []

    for config_cnt, config in enumerate(configurations):
        axarr[config_cnt].set_title(config["label"])
        # axarr.set_title(config["label"])
        for variance in path_loss_variances:
            for payload in payload_sizes:

                config["path_variance"] = variance
                config["payload_size"] = payload

                nodes, agents, env = init_nodes(config=config, agent_to_nodes=load_agents() if config["load"] else None)
                run_nodes(nodes, env, days=config["days"], noma=config["noma"])

                mu, sigma = Node.get_energy_per_byte_stats(nodes, nodes[0].base_station)

                results[variance].append(mu)

            axarr[config_cnt].plot(list(payload_sizes),
                                   list(results[variance]),
                                   marker='o',
                                   linestyle='--',
                                   label=('$\sigma_{dB}$: ' + str(variance) + ' (dB)'),
                                   color=colors[variance])

    results.to_pickle("./results/sim_results")

    axarr.legend()
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

    save_dir = save_dir + f"{file_name}/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    name = save_dir + name + ".png"
    f.savefig(name)
    print(f"\nSaved plot {name}")



    plt.show()

def lload_pickle():
    with open("./results/sim_results.pkl", 'rb') as handle:
        b = pickle.load(handle)
        print(b)

def run_configurations(configurations, file_name, save_to_local, main_theme, progression, rest):

    first_run = True
    f_large, axarr_large = None, None
    f_small, axarr_small = None, None

    file_name_large = file_name
    file_name_small = file_name + "_small"
    file_name_results = file_name + "_histogram"
    file_name_counts = file_name + "_counts"
    file_name_sf_distro = file_name + "_sf_distro"
    file_name_tp_distro = file_name + "_tp_distro"
    file_name_adr_start = file_name + "_adr_start"

    # plots all configs in configurations on one large plot, and duplicates most important data onto a smaller plot
    for config_cnt, config in enumerate(configurations):
        simulation_time = config["days"]*1000*60*60*24
        nodes, agents, env = init_nodes(config=config, agent_to_nodes=load_agents() if config["load"] else None)

        run_nodes(nodes, env, days=config["days"], noma=config["noma"])
        if (config["save"]):
            save_agents(agents)

        # basically number of plots in a column (times 2 for all nodes and average)
        num_categories = len(list(nodes[0].rl_measurements.keys()))

        # setting up plots for histograms of results for DER, EE, etc.
        # 'settings' is a set of labels, the same for each of the historgrams in 'results'
        counts = {}

        for key in nodes[0].counts.keys():
            counts[key] = {}

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

            # setting up plots for histograms of results for DER, EE, etc.
            # 'settings' is a set of labels, the same for each of the historgrams in 'results'
            settings = []
            results = {
                "DER": [],
                "Energy Efficiency": [],
                # "Throughput": []
            }

            num_columns_results = len(results.keys())
            num_rows_results = 1
            # adding 3 additional plots for simulation info at the top, loss and results at the bottom
            f_results, axarr_results = plt.subplots(num_rows_results, num_columns_results,
                                                    figsize=(num_columns_results * 7, num_rows_results * 5),
                                                    sharex=True)

            # for sf in LoRaParameters.SPREADING_FACTORS:
            #     counts["Spreading Factors"][sf] = 0
            # for tp in LoRaParameters.TRANSMISSION_POWERS:
            #     counts["Transmission Power"][tp] = 0

            num_rows_counts = 1
            num_columns_counts = len(counts.keys())
            f_counts, axarr_counts = plt.subplots(num_rows_counts, num_columns_counts,
                                                    figsize=(num_columns_counts * 7, num_rows_counts * 5))

            f_sf_distro, axarr_sf_distro = plt.subplots(1, len(configurations), figsize=(7 * len(configurations), 7))
            major_ticks = np.arange(0, 1001, config["sector_size"])
            for ax in axarr_sf_distro:
                ax.set_xlim([0, Config.CELL_SIZE])
                ax.set_ylim([0, Config.CELL_SIZE])
                ax.set_xticks(major_ticks)
                ax.set_yticks(major_ticks)
                ax.grid(True)

            f_tp_distro, axarr_tp_distro = plt.subplots(1, len(configurations), figsize=(7 * len(configurations), 7))
            major_ticks = np.arange(0, 1001, config["sector_size"])
            for ax in axarr_tp_distro:
                ax.set_xlim([0, Config.CELL_SIZE])
                ax.set_ylim([0, Config.CELL_SIZE])
                ax.set_xticks(major_ticks)
                ax.set_yticks(major_ticks)
                ax.grid(True)

            f_adr_start, axarr_adr_start = plt.subplots(1, len(configurations), figsize=(7 * len(configurations), 7))

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

        # Two sets of plots to indicate when during the simulation nodes had their history collected to a point where
        # network started the execution of ADR

        axarr_large[-2][config_cnt].set_xlabel("Time")
        axarr_large[-2][config_cnt].set_ylabel("ADR enabling point")
        axarr_large[-2][config_cnt].set_title(config["title"])
        axarr_adr_start[config_cnt].set_xlabel("Time")
        axarr_adr_start[config_cnt].set_ylabel("ADR enabling point")
        axarr_adr_start[config_cnt].set_title(config["title"])

        # Setting titles for the distribution plots
        axarr_sf_distro[config_cnt].set_title(config["label"].upper())
        axarr_tp_distro[config_cnt].set_title(config["label"].upper()   )

        # TODO First simply plotting all the nodes at the same time, while also recording data for averaging later
        for node in nodes:

            axarr_sf_distro[config_cnt].scatter(node.location.x, node.location.y,
                                                color=LoRaParameters.SPREADING_FACTORS_COLOURS[node.lora_param.sf],
                                                label=LoRaParameters.SPREADING_FACTORS_LABELS[node.lora_param.sf])
            axarr_tp_distro[config_cnt].scatter(node.location.x, node.location.y,
                                                color=LoRaParameters.TRANSMISSION_POWERS_COLOURS[node.lora_param.tp],
                                                label=LoRaParameters.TRANSMISSION_POWERS_LABELS[node.lora_param.tp])

            axarr_adr_start[config_cnt].plot([node.adr_start_time], [1], marker='o')
            axarr_large[-2][config_cnt].plot([node.adr_start_time], [1], marker='o')

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

        # Enabling legend for distribution scatter plots after plotting is done

        def legend_for_distros(plot):
            handles, labels = plot.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plot.legend(by_label.values(), by_label.keys())

        legend_for_distros(axarr_sf_distro[config_cnt])
        legend_for_distros(axarr_tp_distro[config_cnt])

        # TODO Now plotting the mean values for above
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
                axarr_small[parameter_cnt].set_title(config["title"])
                axarr_small[parameter_cnt].legend()

        # TODO Plotting losses for deep learning agents
        agent_id = 0
        for agent in agents:
            losses = agent.losses

            if (len(configurations) == 1):
                axarr_large[-1].plot(list(losses.keys()), list(losses.values()), label=agent_id)
                agent_id += 1
                axarr_large[-1].set_xlabel("Time")
                axarr_large[-1].set_ylabel("Loss per Agent")
                axarr_large[-1].set_title(config["title"])
            else:
                axarr_large[-1][config_cnt].plot(list(losses.keys()), list(losses.values()), label=agent_id)
                agent_id += 1
                axarr_large[-1][config_cnt].set_xlabel("Time")
                axarr_large[-1][config_cnt].set_ylabel("Loss per Agent")
                axarr_large[-1][config_cnt].set_title(config["title"])

        # TODO record information for current simulation run for the final histograms
        def simulation_results():

            temp = []
            total_unique_packets_sent = 0
            energy_avg = 0

            for node in nodes:
                energy_avg += node.total_energy_consumed()
                throughputs = node.rl_measurements["throughputs"]
                temp.append(list(throughputs.values())[-1])
                total_unique_packets_sent += node.num_unique_packets_sent

                # Recording counts for specified parameters (in Node class)
                for parameter in counts.keys():
                    for val, cnt in node.counts[parameter].items():
                        if (val not in counts[parameter]):
                            counts[parameter][val] = 0
                        counts[parameter][val] += cnt

            # plotting counts values
            for parameter_counter, parameter in enumerate(counts):
                count_dict = counts[parameter]

                x_axis = [i for i in range(len(count_dict))]
                x_axis_labels = [std_val for std_val in count_dict.keys()]
                y_axis = [cnt_val for cnt_val in count_dict.values()]

                # for num, std_parameter_value in enumerate(count_dict):
                #     y_axis.append(count_dict[std_parameter_value])
                # cant think of the index for the current plot, the index is the index of the parameter
                axarr_counts[parameter_counter].plot(x_axis, y_axis, label=config["label"], marker='o')
                axarr_counts[parameter_counter].xaxis.set_ticks(x_axis)
                axarr_counts[parameter_counter].xaxis.set_ticklabels(x_axis_labels)
                axarr_counts[parameter_counter].set_xlabel(parameter)
                axarr_counts[parameter_counter].set_ylabel(parameter + " Counts")
                axarr_counts[parameter_counter].legend()

            # further division by a thousand is required for converting from mJ to J
            energy_avg = (energy_avg / config["num_nodes"]) / 1000
            num_packets_avg = total_unique_packets_sent / config["num_nodes"]

            total_unique_packets_received = nodes[0].base_station.distinct_packets_received
            mean_throughput = np.mean(temp)
            # energy_efficiency = num_packets_avg / energy_avg
            energy_efficiency = mean_throughput / energy_avg
            print(f"TOTAL RECEIVED = {total_unique_packets_received}")
            print(f"TOTAL SENT = {total_unique_packets_sent}")
            der = (total_unique_packets_received / total_unique_packets_sent) * 100

            def print_values_obsolete():
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
                    axarr_large[-1][config_cnt].text(0.1, 0.9, simulation_results, fontsize=20, bbox=props,
                                                     verticalalignment='top',
                                                     transform=axarr_large[-1][config_cnt].transAxes)

            settings.append(config["label"])
            results["DER"].append(der)
            results["Energy Efficiency"].append(energy_efficiency)
            # results["Throughput"].append(mean_throughput)

        simulation_results()

    # plot for histograms at the end
    def plot_results():
        for cnt, val in enumerate(results):
            # axarr_results[0][cnt].ylabel("Energy Efficiency (Packets/Joule)")
            ret = axarr_results[cnt].bar(settings, results[val], width=0.4, color='lightblue')
            axarr_results[cnt].set_ylabel(val)
            axarr_results[cnt].set_title(val)
            # for tick in axarr_results[cnt].get_xticklabels():
            #     tick.set_rotation(-45)
            # for i, val in enumerate(results[val]):
            #     axarr_results[cnt].text(i, val + 1, str("%.2f" % val), color='black', fontweight='bold')
            ret.data_values = [float("%.2f" % v) for v in ret.datavalues]
            axarr_results[cnt].bar_label(ret, label_type='center', fontweight='bold', fontsize=10)

    plot_results()

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

        save_dir = save_dir + f"{file_name_large}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        name = save_dir + file_name_large + ".png"
        f_large.savefig(name)
        print(f"\nSaved plot {name}")

        name = save_dir + file_name_small + ".png"
        f_small.savefig(name)
        print(f"Saved plot {name}")

        name = save_dir + file_name_results + ".png"
        f_results.savefig(name)
        print(f"Saved plot {name}")

        name = save_dir + file_name_counts + ".png"
        f_counts.savefig(name)
        print(f"Saved plot {name}")

        name = save_dir + file_name_sf_distro + ".png"
        f_sf_distro.savefig(name)
        print(f"Saved plot {name}")

        name = save_dir + file_name_tp_distro + ".png"
        f_tp_distro.savefig(name)
        print(f"Saved plot {name}")

        name = save_dir + file_name_adr_start + ".png"
        f_adr_start.savefig(name)
        print(f"Saved plot {name}")

        name = save_dir + file_name_large + ".txt"
        with open(name, 'w') as f:

            def dict_to_str(dict, depth=0):
                ret = "{"
                for key, val in dict.items():
                    tabs = "\t" + (depth * "\t")
                    val_str = val if isinstance(val, bool) else f"\"{val}\"" if isinstance(val, str) else f"{val}"
                    ret += f"\n{tabs}\"{key}\": {val_str},"
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
    # save_figures()
    plt.show()

# Standard configuration values
num_nodes = 10
locations = list()
for i in range(num_nodes):
    locations.append(Location(min=0, max=Config.CELL_SIZE, indoor=False))

normal_reward = "normal"
thoughput_reward = "throughput"
energy_reward = "energy"

def generate_config(config):
    standard_body = {
        "toy_log": False,
        "toy_logg": False,
        "title": "",
        "file_name": "",
        "label": "",
        "conf": False,
        "adr": False,
        "training": False,
        "deep": False,
        "depth": 2,
        "width": 100,
        "sarsa": False,
        "mc": False,
        "replay_buffer": False,
        "replay_buffer_scale": 1,
        "double_deep": False,
        "target_update_rate": 1,
        "load": False,
        "save": False,
        "num_nodes": 0,
        "reward": normal_reward,
        "days": 10,
        "locations": locations,
        "static_locations": True,
        "state_space": ["tp", "sf", "channel"],
        "sector_size": 100,
        "gamma": 0.5,
        "slow_sf": False,
        "slow_tp": False,
        "slow_channel": False,
        "noma": False,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "GLIE": False,
        "Robbins-Monroe": False,
        "epsilon_decay_rate": -1,
        "alpha_decay_rate": -1,
        "expected_sarsa": False,
        "path_variance": 7.8,
        "payload_size": 25,
    }

    for (key, val) in config.items():
        standard_body[key] = val

    return standard_body

# Hypertuning progression
progression = [
    {
        "label": "10",
        "num_nodes": 1,
    },
    {
        "label": "100",
        "num_nodes": 1,
    },
    # {
    #     "label": "1000",
    #     "num_nodes": 1000,
    # },
    # {
    #     "label": "10000",
    #     "num_nodes": 10000,
    # },
]

config_global = [
    [
        "scalability",
        {
            "title": "Scalability",
            "days": 2,
            "static_locations": False,
            # "training": True,
            # "deep": True,
            # "noma": True,
            # "GLIE": True,
            # "Robbins-Monroe": True,
            # "epsilon_decay_rate": 1,
            # "alpha_decay_rate": 0.5,
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
        # title = main_theme["title"] + ", " + temp["title"]
        title = main_theme["title"]
        for (key, val) in main_theme.items():
            temp[key] = val
        temp["title"] = title

        temp = generate_config(temp)
        configs_combination_per_simulation.append(temp)

    print(f"\n\n\n STARTING SIMULATION: \t\t\t {simulation_name} \n\n\n")
    # try:
    # health_check(configurations=configs_combination_per_simulation, days=0.1)
    run_configurations(configurations=configs_combination_per_simulation,
                       save_to_local=False,
                       file_name=config_global[i][0],
                       main_theme=main_theme,
                       progression=progression,
                       rest=configs_combination_per_simulation[0])

    # energy_plots(configurations=configs_combination_per_simulation,
    #              file_name=config_global[i][0],
    #              save_to_local=False)

    # except Exception as e:
    #     print(f"THERE WAS A PROBLEM RUNNING SIMULATION — {config_global[i][0]}")
    #     print(f"THERE WAS A PROBLEM RUNNING SIMULATION — {config_global[i][0]}")
    #     print(f"THERE WAS A PROBLEM RUNNING SIMULATION — {config_global[i][0]}")
    #     print(e)

    print(
        "\n\n\n#################################################################################################\n\n\n")


