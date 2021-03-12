import math

import numpy as np
import AirInterface

class SINRModel:
    def __init__(self):

        self.noise = -80  # mean_mean_values
        self.std_noise = 6  # mean_std_values

        # noise floor according to https://www.semtech.com/uploads/documents/an1200.22.pdf
        self.noise_floor = -174 + 10 * np.log10(125e3)

    def rss_to_sinr(self, rss: float, total_power):
        noise_term = 0.0
        # print(f"rss == {rss}")
        # print(f"total == {total_power}")
        return rss / ((total_power) + self.noise_floor)

    def sinr_to_throughput(self, sinr: float):
        return math.log(sinr + 1, 2)

def roundup(x, GRID_SIZE):
    x = np.divide(x, GRID_SIZE)
    return np.ceil(x).astype(int) * GRID_SIZE
