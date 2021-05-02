import math

import numpy as np

class SINRModel:
    def __init__(self):

        self.noise = -80  # mean_mean_values
        self.std_noise = 6  # mean_std_values

        # noise floor according to https://www.semtech.com/uploads/documents/an1200.22.pdf
        self.noise_floor = -174 + 10 * np.log10(125e3)

    def rss_to_sinr(self, rss: float, total_power):
        # here rss, total_power and noise_floor are all measured in dB
        # converting x to dB => x = 10 * log10(x)
        # converting x from dB => x = 10 ** (x / 10)
        sinr_db = rss - 10 * np.log10(10 ** (total_power / 10) + 10 ** (self.noise_floor / 10))
        sinr = 10 ** (sinr_db / 10)
        return sinr

    def sinr_to_throughput(self, sinr: float):
        return math.log(sinr + 1, 2)

def roundup(x, GRID_SIZE):
    x = np.divide(x, GRID_SIZE)
    return np.ceil(x).astype(int) * GRID_SIZE
