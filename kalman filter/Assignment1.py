import numpy as np
from sim.sim1d import sim_run

# Simulator options
options = {}
options['FIG_SIZE'] = [8, 8]
options['CONSTANT_SPEED'] = True    # False speed not constant

class KalmanFilterToy:
    def __init__(self):
        self.v = 0
        self.prev_x = 0
        self.prev_t = 0

    def predict(self, t):
        prediction = 0       # could change the prediction and speed
        #prediction = 30
        #self.v = 2
        return prediction

    def measure_and_update(self, x, t):
        return


sim_run(options, KalmanFilterToy)