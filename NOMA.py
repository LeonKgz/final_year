from Gateway import Gateway
import AirInterface

class NOMA:
    def __init__(self, gateway: Gateway, air_interface: AirInterface, env, config):
        self.gateway = gateway
        self.air_interface = air_interface
        self.env = env
        self.config = config

    def run(self):
        # Constantly execute the noma algorithm without timeouts TODO: check whether timeouts are necessary
        while True:
            if (len(self.air_interface.packages_in_air_to_noma) > 0):
                # execute on the last element
                to_process = self.air_interface.packages_in_air_to_noma[-1]
                # TODO: potentially need yield self.env.process here
                self.air_interface.packages_in_air_to_noma.remove(to_process)
                if (self.config["toy_log"]):
                    print(f"TOY_NOMA: ################ NOMA: calling self.gateway.packet_received_noma({to_process.node.id}-{to_process.id})")
                yield self.env.process(self.gateway.packet_received_noma(to_process.node, to_process, self.env.now))
                # self.gateway.packet_received_noma(to_process.node, to_process, self.env.now)
            else:
                # if (self.config["toy_log"]):
                #     print(f"TOY_NOMA: ################ NOMA: queue is empty, going to sleep...")
                yield self.env.timeout(10000)
