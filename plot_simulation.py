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
from SNRModel import SNRModel

class Simulation:

    def __init__(self, adr, conf, colour, label):

        # self.transmission_rate = 1e-6  # number of bits sent per ms
        self.transmission_rate = 0.02e-3  # number of bits sent per ms
        self.num_nodes = 100
        self.cell_size = 1000
        # adr = True
        self.adr = adr
        confirmed_messages = True
        self.confirmed_messages = conf

        self.colour = colour
        self.label=label

        self.tx_power_mW = {2: 91.8, 5: 95.9, 8: 101.6, 11: 120.8, 14: 146.5}  # measured TX power for each possible TP
        self.middle = np.round(Config.CELL_SIZE / 2)
        middle = np.round(Config.CELL_SIZE / 2)
        self.gateway_location = Location(x=middle, y=middle, indoor=False)

        self.simulation_num_of_times = 10
        self.mean_energy_per_bit_list = dict()
        self.mean_der = dict()
        self.num_collided_per_node = dict()
        self.num_no_down_per_node = dict()
        self.num_retrans_per_node = dict()
        self.payload_sizes = range(1, 60, 10)

    def plot_time(self, _env):
        while True:
            print('.', end='', flush=True)
            # yield _env.timeout(np.round(Config.SIMULATION_TIME / 10))
            yield _env.timeout(np.round(Config.SIMULATION_TIME / 10))

    def plot(self, fig, ax):
        for payload_size in self.payload_sizes:
            counter = 0

            self.mean_energy_per_bit_list[payload_size] = 0
            self.mean_der[payload_size] = 0
            self.num_collided_per_node[payload_size] = 0
            self.num_no_down_per_node[payload_size] = 0
            self.num_retrans_per_node[payload_size] = 0

            for num_of_times in range(self.simulation_num_of_times):
                env = simpy.Environment()
                gateway = Gateway(env, self.gateway_location)
                nodes = []
                air_interface = AirInterface(gateway, PropagationModel.LogShadow(), SNRModel(), env)

                for node_id in range(self.num_nodes):
                    location = Location(min=0, max=self.cell_size, indoor=False)
                    energy_profile = EnergyProfile(5.7e-3, 15, self.tx_power_mW,
                                                   # rx_power={'pre_mW': 8.2, 'pre_ms': 3.4, 'rx_lna_on_mW': 39,
                                                   #           'rx_lna_off_mW': 34,
                                                   #           'post_mW': 8.3, 'post_ms': 10.7})
                                                   rx_power={'pre_mW': 8.2, 'pre_ms': 3.4, 'rx_lna_on_mW': 36.96,
                                                               'rx_lna_off_mW': 34.65,
                                                               'post_mW': 8.3, 'post_ms': 10.7})
                    lora_param = LoRaParameters(freq=np.random.choice(LoRaParameters.DEFAULT_CHANNELS),
                                                # sf=np.random.choice(LoRaParameters.SPREADING_FACTORS),
                                                sf=9,
                                                bw=125, cr=5, crc_enabled=1, de_enabled=0, header_implicit_mode=0, tp=14)
                    node = Node(node_id, energy_profile, lora_param, payload_size * 8 / self.transmission_rate, process_time=5,
                                          adr=self.adr,
                                location=location,
                                base_station=gateway, env=env, payload_size=payload_size, air_interface=air_interface, confirmed_messages=self.confirmed_messages)
                    nodes.append(node)
                    env.process(node.run())

                env.process(self.plot_time(env))

                d = datetime.timedelta(milliseconds=Config.SIMULATION_TIME)
                print('Running simulator for {}.'.format(d))
                env.run(until=Config.SIMULATION_TIME)
                counter += 1
                print('Simulator is done for payload size {}; counter={}'.format(payload_size, counter))
                mean_energy_per_bit = 0
                num_collided = 0
                num_retrans = 0
                num_no_down = 0
                for node in nodes:
                    mean_energy_per_bit += node.transmit_related_energy_per_bit()
                    num_collided += node.num_collided
                    num_retrans += node.num_retransmission
                    num_no_down += node.num_no_downlink

                self.num_collided_per_node[payload_size] += num_collided / self.num_nodes
                self.num_no_down_per_node[payload_size] += num_no_down / self.num_nodes
                self.num_retrans_per_node[payload_size] += num_retrans / self.num_nodes

                self.mean_energy_per_bit_list[payload_size] += mean_energy_per_bit / self.num_nodes

                der = gateway.get_der(nodes)
                self.mean_der[payload_size] += der

            self.mean_energy_per_bit_list[payload_size] = self.mean_energy_per_bit_list[payload_size] / self.simulation_num_of_times
            self.mean_der[payload_size] = self.mean_der[payload_size] / self.simulation_num_of_times
            self.num_collided_per_node[payload_size] = self.num_collided_per_node[payload_size] / self.simulation_num_of_times
            self.num_no_down_per_node[payload_size] = self.num_no_down_per_node[payload_size] / self.simulation_num_of_times
            self.num_retrans_per_node[payload_size] = self.num_retrans_per_node[payload_size] / self.simulation_num_of_times

        # Printing experiment parameters
        print('{} nodes in network'.format(self.num_nodes))
        print('{} transmission rate'.format(self.transmission_rate))
        print('{} ADR'.format(self.adr))
        print('{} confirmed msgs'.format(self.confirmed_messages))
        print('{}m cell size'.format(self.cell_size))


        # fig1 = plt.subplot(2, 1, 1)

        ax.set_xlabel('Payload Sizes')
        ax.set_ylabel('Energy per bit')
        return plt.plot(self.payload_sizes, list(self.mean_energy_per_bit_list.values()), self.colour, label=self.label, linestyle='-.')
        # fig2 = plt.subplot(2, 1, 2)

        # # plt.xscale('log', basex=2)
        # plt.subplot(6, 1, 2)
        # plt.plot(payload_sizes, list(mean_der.values()))
        # # plt.xscale('log', basex=2)
        # plt.subplot(6, 1, 3)
        # plt.plot(payload_sizes, np.divide(list(mean_energy_per_bit_list.values()), list(mean_der.values())))
        # # plt.xscale('log', basex=2)
        # plt.subplot(6, 1, 4)
        # plt.plot(payload_sizes, list(num_no_down_per_node.values()))
        # # plt.xscale('log', basex=2)
        # plt.subplot(6, 1, 5)
        # plt.plot(payload_sizes, list(num_collided_per_node.values()))
        # # plt.xscale('log', basex=2)
        # plt.subplot(6, 1, 6)
        # plt.plot(payload_sizes, list(num_retrans_per_node.values()))
        # # plt.xscale('log', basex=2)


fig, ax = plt.subplots()

sim1 = Simulation(adr=True, conf=True, colour='g', label='ADR CONF')
sim2 = Simulation(adr=False, conf=False, colour='b', label='NO ADR NO CONF')
sim3 = Simulation(adr=True, conf=False, colour='orange', label='ADR NO CONF')

l1, = sim1.plot(fig, ax)
print("--------------------SIMULATION 1 IS DONE ----------------")
l2, = sim2.plot(fig, ax)
print("--------------------SIMULATION 2 IS DONE ----------------")
l3, = sim3.plot(fig, ax)
print("--------------------SIMULATION 3 IS DONE ----------------")

plt.legend(handles=[l1, l2, l3])
plt.show()
