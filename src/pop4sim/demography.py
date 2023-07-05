import numpy as np
from scipy.interpolate import interp1d

__author__ = 'Chu-Chang Ku'
__all__ = ['Demography']


def interp_cont(years, v):
    return interp1d(years, v, axis=0, kind='linear', bounds_error=False, fill_value=(v[0], v[-1]))


def interp_dis(years, v):
    return interp1d(years, v, axis=0, kind='nearest-up', bounds_error=False, fill_value=(v[0], v[-1]))


class Demography:
    def __init__(self, src, ty='cont'):
        self.DimNames = src['dimnames']
        self.Source = dict(src)
        self.Years = years = src['Year']
        self.YearSpan = min(years), max(years)

        self.Type = ty
        fn = interp_cont if ty == 'cont' else interp_dis

        self.fn = fn
        self.N = fn(years, src['N'])
        self.RateBirth = fn(years, src['RateBirth'])
        self.RateDeath = fn(years, src['RateDeath'])
        self.RateAgeing = fn(years, src['RateAgeing'])
        self.RateMig = None

    def append_mig(self, mig):
        fn = interp_cont if self.Type == 'cont' else interp_dis
        self.Source['RateMig'] = mig
        self.RateMig = fn(self.Years, mig)

    def calc_mig(self, t, y):
        n = self.N(t)
        return 50 * (n - y) / n

    def __call__(self, t, y=None):
        pars = {
            'N': self.N(t),
            'r_birth': self.RateBirth(t),
            'r_ageing': self.RateAgeing(t),
            'r_death': self.RateDeath(t)
        }
        if self.RateMig is not None:
            pars['r_mig'] = self.RateMig(t)
        elif y is not None:
            pars['r_mig'] = self.calc_mig(t, y)
        else:
            pars['r_mig'] = np.zeros_like(pars['r_death'])

        return pars