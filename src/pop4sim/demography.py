import numpy as np
from scipy.interpolate import interp1d

__author__ = 'Chu-Chang Ku'
__all__ = ['Demography']


class Demography:
    def __init__(self, ext):
        self.Inputs = dict()
        self.Years = years = ext['Year']
        for k, v in ext.items():
            if k not in ['Year', 'Age']:
                self.Inputs[k] = interp1d(years, v, axis=0, bounds_error=False, fill_value=(v[0], v[-1]))

        self.YearSpan = [np.min(years), np.max(years)]

    def __call__(self, t):
        return {k: v(t) for k, v in self.Inputs.items()}
