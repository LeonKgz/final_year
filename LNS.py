from GatewayMultiple import Gateway2
from LoRaPacket import UplinkMessage

class LNS:

    def __init__(self, env, location, config):
        self.env = env
        self.location = location
        self.gateways = None
        self.queues = {}
        self.unique_packets_queued = set()
        self.config = config


    def add_gateways(self, gateways):
        self.gateways = gateways
        for g in gateways:
            self.queues[g] = {}

    def submit(self, packet: UplinkMessage, gateway: Gateway2):
        if (self.config["toy_log"]):
            print(f"TOY_NOMA: ################ LNS: received packet {packet.node.id}-{packet.id}")
        self.queues[gateway][packet.id] = packet
        self.unique_packets_queued.add(packet.id)
        # yield self.env.process(gateway.packet_received_noma(packet.node, packet, self.env.now))
        yield self.env.timeout(1)

    # running it like this allows for other gateways time to decode
    # the same packet and now decide which one needs to respond
    # based on packets rss value
    def run(self):
        # Constantly check the state of submitted packets
        while (True):

            if (self.unique_packets_queued):
                # First scan the queues for each gateway, identify optimal if duplicates are present and afterwards
                # clear the queues appropriately

                packets_to_remove = {}
                final_verdict = []

                for g in self.gateways:
                    for packet_id in self.queues[g]:
                        # marking packet to clear after the scan
                        # if the packet is already in the dictionary the respective copies would have been added as well
                        if (packet_id not in packets_to_remove):
                            packets_to_remove[packet_id] = []
                        packets_to_remove[packet_id].append(g)

                for packet_id in self.unique_packets_queued:
                    optimum_gateway = packets_to_remove[packet_id][0]
                    if (len(packets_to_remove[packet_id]) > 1):
                        for g in packets_to_remove[packet_id]:
                           first_value = (10 ** self.queues[g][packet_id].rss) / 10
                           second_value = (10 ** self.queues[optimum_gateway][packet_id].rss) / 10
                           if first_value > second_value:
                               optimum_gateway = g
                    final_verdict.append((optimum_gateway, self.queues[optimum_gateway][packet_id]))

                # Schedule downlink messages for gateways
                for gateway, packet in final_verdict:
                    if (self.config["toy_log"]):
                        print(f"TOY_NOMA: ################ LNS: sending packet {packet.node.id}-{packet.id} to GATEWAY {gateway}")
                    yield self.env.process(gateway.packet_received_noma(packet.node, packet, self.env.now))
                    # self.env.process(gateway.packet_received_noma(packet.node, packet, self.env.now))
                    # gateway.packet_received_noma(packet.node, packet, self.env.now)

                # Clear the queues
                for id, gs in packets_to_remove.items():
                    if id in self.unique_packets_queued:
                        self.unique_packets_queued.remove(id)
                    for g in gs:
                        del self.queues[g][id]

            else:
                # if (self.config["toy_log"]):
                    # print(f"TOY_NOMA: ################ LNS: queues are empty, sleeping...")
                yield self.env.timeout(100)
