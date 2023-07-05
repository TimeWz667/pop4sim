import numpy as np
from scipy.integrate import solve_ivp
from pop4sim.demography import Demography
from pop4sim.fetcher import RawData

__author__ = 'Chu-Chang Ku'
__all__ = ['ModelAll', 'reform_pars_all']


class ModelAll:
    def __init__(self, demo, cohort=False):
        self.IsCohort = cohort
        self.Demography = demo

    def __call__(self, t, y):
        pars = self.Demography(t)
        dy = np.zeros_like(y)

        deaths = pars['DeaR'] * y
        births = pars['BirR'] * y.sum()

        dy -= deaths

        if self.IsCohort:
            return dy

        dy += births

        if 'MigR' in pars:
            mig = pars['MigR'] * y
        else:
            mig = 10 * (pars['N'] - y) / pars['N'] * y
        dy += mig

        return dy.reshape(-1)

    def calc_mig(self, t, y):
        pars = self.Demography(t)
        return 10 * (pars['N'] - y) / pars['N']

    def get_y0(self, t):
        pars = self.Demography(t)
        return pars['N'].copy()


def reform_pars_all(ext: RawData, mig=False):
    ns = (ext.Population['F'] + ext.Population['M']).sum(1)
    dea = (ext.Population['F'] * ext.RateDeath['F'] + ext.Population['M'] * ext.RateDeath['M']).sum(1)
    stacked = {
        'N': ns,
        'DeaR': dea / ns,
        'BirR': ext.RateBirth['F'] + ext.RateBirth['M'],
        'Year': ext.Years,
    }

    demo = Demography(stacked)

    if not mig:
        return demo, stacked

    model = ModelAll(demo)

    year = ext.Years
    t_span = ext.YearSpan
    n_yr = len(year)
    y0 = model.get_y0(year[0])
    sol = solve_ivp(model, y0=y0.reshape(-1), t_span=t_span, dense_output=True)

    # Get model-based migration rates
    mig = list()
    for t in year:
        y = sol.sol(t)
        mig.append(model.calc_mig(t, y))

    mig = np.array(mig)

    for i, t in enumerate(year[1:-1]):
        pars = demo(t)
        dp = np.log(demo(t + 0.5)['N']) - np.log(demo(t - 0.5)['N'])
        mig[i + 1] = dp - pars['BirR'] + pars['DeaR']

    t = year[-1]
    pars = demo(t)
    dp = np.log(pars['N']) - np.log(demo(t - 0.5)['N'])
    mig[-1] = dp * 2 - pars['BirR'] + pars['DeaR']

    stacked_mig = dict(stacked)
    stacked_mig['MigR'] = mig
    demo_mig = Demography(stacked_mig)
    return demo_mig, stacked_mig
