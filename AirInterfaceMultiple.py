import random

import Global
import PropagationModel
from GatewayMultiple import Gateway2
from LNS import LNS
from Location import Location
from Gateway import Gateway
from LoRaPacket import UplinkMessage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from NodeMultiple import Node2
from SNRModel import SNRModel
from SINRModel import SINRModel


class AirInterface2:
    def __init__(self, lns: LNS, gateways, prop_model: PropagationModel, snr_model: SNRModel, sinr_model: SINRModel, env, config):

        self.prop_measurements = {}
        self.num_of_packets_collided = {}
        self.num_of_packets_send = 0
        self.lns = lns
        self.gateways = gateways
        for gateway in gateways:
            gateway.air_interface = self
            gateway.noma.air_interface = self

        self.packages_in_air = {}
        self.packages_in_air_to_noma = {}

        for g in gateways:
            self.packages_in_air[g] = []
            self.packages_in_air_to_noma[g] = []
            self.num_of_packets_collided[g] = 0

        self.color_per_node = dict()
        self.prop_model = prop_model
        self.snr_model = snr_model
        self.sinr_model = sinr_model
        self.env = env

        self.config = config

    @staticmethod
    def frequency_collision(p1: UplinkMessage, p2: UplinkMessage):
        """frequencyCollision, conditions
                |f1-f2| <= 120 kHz if f1 or f2 has bw 500
                |f1-f2| <= 60 kHz if f1 or f2 has bw 250
                |f1-f2| <= 30 kHz if f1 or f2 has bw 125
        """

        p1_freq = p1.lora_param.freq
        p2_freq = p2.lora_param.freq

        p1_bw = p1.lora_param.bw
        p2_bw = p2.lora_param.bw

        if abs(p1_freq - p2_freq) <= 120 and (p1_bw == 500 or p2_bw == 500):
            if Global.Config.PRINT_ENABLED:
                print("frequency coll 500")
            return True
        elif abs(p1_freq - p2_freq) <= 60 and (p1_bw == 250 or p2_bw == 250):
            if Global.Config.PRINT_ENABLED:
                print("frequency coll 250")
            return True
        elif abs(p1_freq - p2_freq) <= 30 and (p1_bw == 125 or p2_bw == 125):
            if Global.Config.PRINT_ENABLED:
                print("frequency coll 125")
            return True

        if Global.Config.PRINT_ENABLED:
            print("no frequency coll")
        return False

    @staticmethod
    def sf_collision(p1: UplinkMessage, p2: UplinkMessage):
        #
        # sfCollision, conditions
        #
        #       sf1 == sf2
        #
        if p1.lora_param.sf == p2.lora_param.sf:
            if Global.Config.PRINT_ENABLED:
                print("collision sf node {} and node {}".format(p1.node.id, p2.node.id))
            return True
        if Global.Config.PRINT_ENABLED:
            print("no sf collision")
        return False

    @staticmethod
    def timing_collision(me: UplinkMessage, other: UplinkMessage):
        # packet p1 collides with packet p2 when it overlaps in its critical section

        sym_duration = 2 ** me.lora_param.sf / (1.0 * me.lora_param.bw)
        num_preamble = 8
        critical_section_start = me.start_on_air + sym_duration * (num_preamble - 5)
        critical_section_end = me.start_on_air + me.my_time_on_air()

        if Global.Config.PRINT_ENABLED:
            print('P1 has a critical section in [{} - {}]'.format(critical_section_start, critical_section_end))

        other_end = other.start_on_air + other.my_time_on_air()

        if other_end < critical_section_start or other.start_on_air > critical_section_end:
            # all good
            me_time_collided = False
        else:
            # timing collision
            me_time_collided = True

        sym_duration = 2 ** other.lora_param.sf / (1.0 * other.lora_param.bw)
        num_preamble = 8
        critical_section_start = other.start_on_air + sym_duration * (num_preamble - 5)
        critical_section_end = other.start_on_air + other.my_time_on_air()

        if Global.Config.PRINT_ENABLED:
            print('P2 has a critical section in [{} - {}]'.format(critical_section_start, critical_section_end))

        me_end = me.start_on_air + me.my_time_on_air()

        if me_end < critical_section_start or me.start_on_air > critical_section_end:
            # all good
            other_time_collided = False
        else:
            # timing collision
            other_time_collided = True

        # return who has been time collided (can be with each other)
        if me_time_collided and other_time_collided:
            return (me, other)
        elif me_time_collided:
            return (me,)
        elif other_time_collided:
            return (other_time_collided,)
        else:
            return None

    @staticmethod
    def power_collision(me: UplinkMessage, other: UplinkMessage, time_collided_nodes):
        power_threshold = 6  # dB
        if Global.Config.PRINT_ENABLED:
            print(
                "pwr: node {0.node.id} {0.rss:3.2f} dBm node {1.node.id} {1.rss:3.2f} dBm; diff {2:3.2f} dBm".format(me,other,round(me.rss - other.rss,2)))
        if abs(me.rss - other.rss) < power_threshold:
            if Global.Config.PRINT_ENABLED:
                print("collision pwr both node {} and node {} (too close to each other)".format(me.node.id,
                                                                                                other.node.id))
            if me in time_collided_nodes:
                me.collided = True
            if other in time_collided_nodes:
                other.collided = True

        elif me.rss - other.rss < power_threshold:
            # me has been overpowered by other
            # me will collided if also time_collided

            if me in time_collided_nodes:
                if Global.Config.PRINT_ENABLED:
                    print("collision pwr both node {} has collided by node {}".format(me.node.id,other.node.id))
                me.collided = True
        else:
            # other was overpowered by me
            if other in time_collided_nodes:
                if Global.Config.PRINT_ENABLED:
                    print("collision pwr both node {} has collided by node {}".format(other.node.id,me.node.id))
                other.collided = True

    def collision(self, packet, gateway) -> bool:
        if Global.Config.PRINT_ENABLED:
            print("CHECK node {} (sf:{} bw:{} freq:{:.6e}) #others: {}".format(
                packet.node.id, packet.lora_param.sf, packet.lora_param.bw, packet.lora_param.freq,
                len(self.packages_in_air)))
        if packet.collided:
            return True
        for other in self.packages_in_air[gateway]:
            if other.node.id != packet.node.id:
                if Global.Config.PRINT_ENABLED:
                    print(">> node {} (sf:{} bw:{} freq:{:.6e})".format(
                        other.node.id, other.lora_param.sf, other.lora_param.bw,
                        other.lora_param.freq))
                if AirInterface2.frequency_collision(packet, other):
                    if AirInterface2.sf_collision(packet, other):
                        time_collided_nodes = AirInterface2.timing_collision(packet, other)
                        if time_collided_nodes is not None:
                            AirInterface2.power_collision(packet, other, time_collided_nodes)
        return packet.collided

    color_values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

    def get_gateways_within_reach(self, node: Node2):
        # xs = [g.location.x for g in self.gateways]
        # ys = [g.location.y for g in self.gateways]
        return list(filter(lambda g: Location.distance(g.location, node.location) <= node.reach, self.gateways))

    def get_gateways_within_reach_simple(self, node: Node2):
        ret = []
        for g in self.gateways:
            if (Location.distance(g.location, node.location) <= node.reach):
                ret.append(g)
        return ret

    def submit(self, packet: UplinkMessage):
        # available_gateways = self.get_gateways_within_reach(packet.node)
        available_gateways = self.get_gateways_within_reach_simple(packet.node)

        # return self.packet_in_air_past_noma(packet, available_gateways)
        return self.packet_in_air_through_noma(packet, available_gateways)

    def packet_in_air_through_noma(self, packet: UplinkMessage, gateways):

        self.num_of_packets_send += 1
        id = packet.node.id

        if id not in self.color_per_node:
            self.color_per_node[id] = '#' + random.choice(AirInterface2.color_values) + random.choice(
                AirInterface2.color_values) + random.choice(AirInterface2.color_values) + random.choice(
                AirInterface2.color_values) + random.choice(AirInterface2.color_values) + random.choice(
                AirInterface2.color_values)

        node_id = packet.node.id
        from_node = packet.node

        for gateway in gateways:
            rss = self.prop_model.tp_to_rss(from_node.location.indoor, packet.lora_param.tp,
                                            Location.distance(gateway.location, packet.node.location))

            if gateway not in self.prop_measurements:
                self.prop_measurements[gateway] = {}

            if node_id not in self.prop_measurements[gateway]:
                self.prop_measurements[gateway][node_id] = {'rss': [], 'snr': [], 'sinr': [], 'throughput': [], 'time': [],
                                                   'pkgs_in_air': [], 'time_for_pkgs': []}

            packet.rss = rss
            snr = self.snr_model.rss_to_snr(rss)
            packet.snr = snr

            total_power_acc = 0

            if (packet in self.packages_in_air[gateway]):
                raise Exception('packet in question is in packages in air (added below) (AirInterface, line 200)')

            for p in self.packages_in_air[gateway]:
                # convert from dB values
                p_rss = 10 ** (p.rss / 10)
                packet_rss = 10 ** (packet.rss / 10)
                if (p_rss < packet_rss and p.lora_param.freq == packet.lora_param.freq):
                    total_power_acc += p_rss

            np.seterr(divide='ignore')
            # np.seterr(divide='warn')
            total_power_db = 10 * np.log10(total_power_acc)

            # sinr = self.sinr_model.rss_to_sinr(rss, sum(filter(lambda x: x < rss, map(lambda x: x.rss, filter(lambda x: x.lora_param.freq == packet.lora_param.freq, self.packages_in_air)))))
            # sinr = self.sinr_model.rss_to_sinr(rss, sum(filter(lambda x: x < rss, map(lambda x: x.rss, self.packages_in_air))))

            sinr = self.sinr_model.rss_to_sinr(rss, total_power_db)
            packet.sinr = sinr
            throughput = self.sinr_model.sinr_to_throughput(sinr)

            ##########################################################################

            # self.packages_in_air_to_noma[gateway].append(packet)

            self.prop_measurements[gateway][node_id]['time'].append(self.env.now)
            self.prop_measurements[gateway][node_id]['rss'].append(rss)
            self.prop_measurements[gateway][node_id]['snr'].append(snr)
            self.prop_measurements[gateway][node_id]['sinr'].append(sinr)
            self.prop_measurements[gateway][node_id]['throughput'].append(throughput)
            self.prop_measurements[gateway][node_id]['pkgs_in_air'].append(len(self.packages_in_air))
            self.prop_measurements[gateway][node_id]['time_for_pkgs'].append(self.env.now)

            self.packages_in_air[gateway].append(packet)

        return gateways

    # def packet_in_air_past_noma(self, packet: UplinkMessage, gateways):
    #
    #     self.num_of_packets_send += 1
    #     id = packet.node.id
    #
    #     if id not in self.color_per_node:
    #         self.color_per_node[id] = '#' + random.choice(AirInterface2.color_values) + random.choice(
    #             AirInterface2.color_values) + random.choice(AirInterface2.color_values) + random.choice(
    #             AirInterface2.color_values) + random.choice(AirInterface2.color_values) + random.choice(
    #             AirInterface2.color_values)
    #
    #     from_node = packet.node
    #     node_id = from_node.id
    #
    #     max_rss = 0
    #     best_gateway = None
    #
    #     for gateway in gateways:
    #
    #         rss_db = self.prop_model.tp_to_rss(from_node.location.indoor, packet.lora_param.tp,
    #                                         Location.distance(gateway.location, packet.node.location))
    #         rss = 10 ** (rss_db / 10)
    #         if (rss > max_rss):
    #             max_rss = rss
    #             best_gateway = gateway
    #
    #     # Converting best rss value back to decibels
    #     rss = 10 * np.log10(max_rss)
    #
    #     if node_id not in self.prop_measurements:
    #         self.prop_measurements[node_id] = {'rss': [], 'snr': [], 'sinr': [], 'throughput': [], 'time': [], 'pkgs_in_air': [], 'time_for_pkgs': []}
    #     packet.rss = rss
    #     snr = self.snr_model.rss_to_snr(rss)
    #     packet.snr = snr
    #     # TODO remove users from other channels
    #
    #     total_power_acc = 0
    #
    #     if (packet in self.packages_in_air):
    #         raise Exception('packet in question is in packages in air (added below) (AirInterfaceMultiple.py)')
    #
    #     for p in self.packages_in_air[best_gateway]:
    #         # convert from dB values
    #         p_rss = 10 ** (p.rss / 10)
    #         packet_rss = 10 ** (packet.rss / 10)
    #         if (p_rss < packet_rss and p.lora_param.freq == packet.lora_param.freq):
    #            total_power_acc += p_rss
    #
    #     np.seterr(divide='ignore')
    #     # np.seterr(divide='warn')
    #     total_power_db = 10 * np.log10(total_power_acc)
    #
    #     # sinr = self.sinr_model.rss_to_sinr(rss, sum(filter(lambda x: x < rss, map(lambda x: x.rss, filter(lambda x: x.lora_param.freq == packet.lora_param.freq, self.packages_in_air)))))
    #     # sinr = self.sinr_model.rss_to_sinr(rss, sum(filter(lambda x: x < rss, map(lambda x: x.rss, self.packages_in_air))))
    #
    #     sinr = self.sinr_model.rss_to_sinr(rss, total_power_db)
    #     packet.sinr = sinr
    #     throughput = self.sinr_model.sinr_to_throughput(sinr)
    #
    #     self.prop_measurements[best_gateway][node_id]['time'].append(self.env.now)
    #     self.prop_measurements[best_gateway][node_id]['rss'].append(rss)
    #     self.prop_measurements[best_gateway][node_id]['snr'].append(snr)
    #     self.prop_measurements[best_gateway][node_id]['sinr'].append(sinr)
    #     self.prop_measurements[best_gateway][node_id]['throughput'].append(throughput)
    #     self.prop_measurements[best_gateway][node_id]['pkgs_in_air'].append(len(self.packages_in_air))
    #     self.prop_measurements[best_gateway][node_id]['time_for_pkgs'].append(self.env.now)
    #
    #     self.packages_in_air[best_gateway].append(packet)
    #     return best_gateway

    def packet_received(self, packet: UplinkMessage, gateways) -> bool:
        """Packet has fully received by the gateway
            This method checks if this packet has collided
            and remove from in the air
            :return bool (True collided or False not collided)"""

        all_collided = True

        for gateway in gateways:

            collided = self.collision(packet, gateway)
            if collided:
                self.num_of_packets_collided[gateway] += 1
                # print('Our packet has collided')
            self.packages_in_air[gateway].remove(packet)
            self.prop_measurements[gateway][packet.node.id]['pkgs_in_air'].append(len(self.packages_in_air))
            self.prop_measurements[gateway][packet.node.id]['time_for_pkgs'].append(self.env.now)

            if (packet.noma and not collided):
                if (self.config["toy_log"]):
                    print(f"TOY_NOMA: ################ AIR INTERFACE calling self.noma_insert({packet.node.id}-{packet.id})")
                self.noma_insert(packet, gateway)

            if (collided and self.config["toy_log"]):
                print(f"TOY_NOMA: ################ AIR INTERFACE: packet {packet.node.id}-{packet.id}) has COLLIDED")

            all_collided = all_collided and collided

        # even though at this point we don't know which gateway will respond
        # if all_collided=False then we know that at "at least" one gateway the packet did not collide so definitely
        # returned value should be False. Vice versa if all_collided = True then collided = True
        return all_collided

    def noma_insert(self, p_new, gateway: Gateway2):
        if (len(self.packages_in_air_to_noma[gateway]) == 0):
            self.packages_in_air_to_noma[gateway].append(p_new)
        elif (p_new.rss > self.packages_in_air_to_noma[gateway][-1].rss):
            self.packages_in_air_to_noma[gateway].append(p_new)
        else:
            for p in self.packages_in_air_to_noma[gateway]:
                if (p_new.rss < p.rss):
                    index = self.packages_in_air_to_noma[gateway].index(p)
                    self.packages_in_air_to_noma[gateway].insert(index, p_new)
                    break

        if (self.config["toy_log"]):
            print(f"TOY_NOMA: ################ AIR INTERFACE: state of self.packages_in_air_to_noma is ??? {[[p.id for p in self.packages_in_air_to_noma[gateway]] for gateway in self.gateways]}")

    def plot_packets_in_air(self):
        plt.figure()
        ax = plt.gca()
        plt.axis('off')
        ax.grid(False)
        for package in self.packages_in_air:
            node_id = package.node.id
            plt.hlines(package.lora_param.freq, package.start_on_air, package.start_on_air + package.my_time_on_air(),
                       color=self.color_per_node[node_id],
                       linewidth=2.0)
        plt.show()

    def log(self):
        print('Total number of packets in the air {}'.format(self.num_of_packets_send))
        print('Total number of packets collided {} {:2.2f}%'.format(self.num_of_packets_collided,
                                                                    self.num_of_packets_collided * 100 / self.num_of_packets_send))

    def get_prop_measurements(self, node_id):
        return self.prop_measurements[node_id]

    def get_simulation_data(self, name) -> pd.Series:
        series = pd.Series([self.num_of_packets_collided, self.num_of_packets_send], index=['NumberOfPacketsCollided','NumberOfPacketsOnAir'])
        series.name = name
        return series.transpose()
