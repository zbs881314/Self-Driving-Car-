import numpy as np
from sim.sim1d import sim_run


# Simulator options
options = {}
options['FIG_SIZE'] = [8, 8]
options['CONSTANT_SPEED'] = False  # could try to change to True


class KalmanFilterToy:
    def __init__(self):
        self.v = 0
        self.prev_x = 0
        self.prev_time = 0

    def predict(self, t):
        prediction = self.v * (t - self.prev_time) + self.prev_x
        return prediction

    def measure_and_update(self, x, t):
        measured_v = (x - self.prev_x) / (t - self.prev_time)
        self.v += 0.5 * (measured_v - self.v)
        #self.v += 0.1 * (measured_v - self.v)
        #self.v += 0.3 * (measured_v - self.v)
        #self.v += 0.9 * (measured_v - self.v)

        self.prev_x = x
        self.prev_time = t
        return


sim_run(options, KalmanFilterToy)