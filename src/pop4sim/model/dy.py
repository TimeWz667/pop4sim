import numpy as np
from pop4sim.demography import Demography
__author__ = 'Chu-Chang Ku'
__all__ = ['ModelODE']


class ModelODE:
    def __init__(self, demo: Demography, cohort=False):
        self.IsCohort = cohort
        self.Demography = demo
        self.HasAge = 'Age' in demo.DimNames
        self.HasSex = 'Sex' in demo.DimNames

    def __call__(self, t, y):
        pars = self.Demography(t)
        y = y.reshape(pars['N'].shape)
        dy = np.zeros_like(y)

        deaths = pars['r_death'] * y
        dy -= deaths

        if self.HasAge:
            ageing = pars['r_ageing'] * y
            dy[:-1] -= ageing[:-1]
            dy[1:] += ageing[:-1]

        if self.IsCohort:
            return dy

        births = pars['r_birth'] * y.sum()

        if self.HasAge:
            dy[0] += births
        else:
            dy += births

        mig = pars['r_mig'] * y
        dy += mig

        return dy.reshape(-1)

    def get_y0(self, t):
        return self.Demography.N(t).copy()
