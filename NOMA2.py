# import AirInterface2
from Gateway import Gateway
import AirInterface

class NOMA:
    def __init__(self, gateway: Gateway, air_interface: AirInterface, env):
        self.gateway = gateway
        self.air_interface = air_interface
        self.env = env

    def run(self):
        # Constantly execute the noma algorithm without timeouts TODO: check whether timeouts are necessary
        while True:
            if (len(self.air_interface.packages_in_air_to_noma[self.gateway]) > 0):
                # execute on the last element
                to_process = self.air_interface.packages_in_air_to_noma[self.gateway][-1]
                # TODO: potentially need yield self.env.process here
                self.air_interface.packages_in_air_to_noma[self.gateway].remove(to_process)
                yield self.env.process(self.gateway.packet_received_noma(to_process.node, to_process, self.env.now))
            else:
                yield self.env.timeout(10000)

