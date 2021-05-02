import numpy as np


class Config:
    LOG_ENABLED = True
    MAX_DELAY_BEFORE_SLEEP_MS = 500
    SIMULATION_TIME = 1000*60*60*24*10
    PRINT_ENABLED = False
    CELL_SIZE = 1000
    MAX_DELAY_START_PER_NODE_MS = np.round(SIMULATION_TIME / 10)
    num_nodes = 1000
    track_changes = True

class Paper1:
    LOG_ENABLED = True
    MAX_DELAY_BEFORE_SLEEP_MS = 500
    SIMULATION_TIME = 1000 * 60 * 60 * 150
    PRINT_ENABLED = False
    CELL_SIZE = 1500
    MAX_DELAY_START_PER_NODE_MS = np.round(SIMULATION_TIME / 10)
    num_nodes = 30
    track_changes = True

