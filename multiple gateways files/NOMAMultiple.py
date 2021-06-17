class NOMA2:
    def __init__(self, gateway, air_interface, env, config):
        self.gateway = gateway
        self.air_interface = air_interface
        self.env = env
        self.config=config

    # Running it separately in a loop like this
    # makes it more realistic as it is an independednt
    # agent not called by anybody else, it acts periodically
    # independently creating a better more concurrent like
    # simulation of packets incoming at a gateway
    def run(self):
        # Constantly execute the noma algorithm with timeouts
        while True:
            if (len(self.air_interface.packages_in_air_to_noma[self.gateway]) > 0):
                # execute on the last element
                to_process = self.air_interface.packages_in_air_to_noma[self.gateway][-1]
                self.air_interface.packages_in_air_to_noma[self.gateway].remove(to_process)
                # yield self.env.process(self.gateway.packet_received_noma(to_process.node, to_process, self.env.now, self.gateway))
                if (self.config["toy_log"]):
                    print(f"TOY_NOMA: ################ NOMA: calling self.gateway.sibmit_packet_for_consideration({to_process.node.id}-{to_process.id})")
                # self.gateway.submit_packet_for_consideration(to_process)
                yield self.env.process(self.gateway.submit_packet_for_consideration(to_process))
                # yield self.env.process(self.gateway.packet_received_noma(to_process.node, to_process, self.env.now))
            else:
                # if (self.config["toy_log"]):
                    # print(f"TOY_NOMA: ################ NOMA: queue is empty, going to sleep...")
                yield self.env.timeout(10000)
