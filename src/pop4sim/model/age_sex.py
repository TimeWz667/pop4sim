import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from pop4sim.demography import Demography
from pop4sim.fetcher import RawData

__author__ = 'Chu-Chang Ku'
__all__ = ['ModelAgeSex', 'reform_pars_agesex']


class ModelAgeSex:
    def __init__(self, demo, cohort=False):
        self.IsCohort = cohort
        self.Demography = demo

    def __call__(self, t, y):
        y = y.reshape((-1, 2))

        pars = self.Demography(t)
        dy = np.zeros_like(y)

        deaths = pars['DeaR'] * y
        births = pars['BirR'] * y.sum()

        ageing = y[:-1]

        dy -= deaths
        dy[:-1] -= ageing
        dy[1:] += ageing

        if self.IsCohort:
            return dy

        dy[0] += births

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


def reform_pars_agesex(ext: RawData, mig=False, opt_mig=True):
    years = ext.Years
    t_span = ext.YearSpan

    stacked = {
        'N': np.stack([ext.Population['F'], ext.Population['M']], 2),
        'DeaR': np.stack([ext.RateDeath['F'], ext.RateDeath['M']], 2),
        'BirR': np.stack([ext.RateBirth['F'], ext.RateBirth['M']], 1),
        'Year': ext.Years['Year'],
        'Age': range(101)
    }

    demo = Demography(stacked)

    if not mig:
        return demo, stacked

    model = ModelAgeSex(demo)

    n_yr = len(years)
    y0 = model.get_y0(t_span[0])
    sol = solve_ivp(model, y0=y0.reshape(-1), t_span=t_span, dense_output=True)

    # Get model-based migration rates
    mig = list()
    for t in years:
        y = sol.sol(t).reshape((101, 2))
        mig.append(model.calc_mig(t, y))

    mig = np.stack(mig)

    if not opt_mig:
        stacked_mig = dict(stacked)
        stacked_mig['MigR'] = mig
        demo_mig = Demography(stacked_mig)
        return demo_mig, stacked_mig

    stacked_mig = dict(stacked)
    stacked_mig['MigR'] = mig.copy()

    t_start = t_span[0]
    y0 = model.get_y0(t_start)

    for i in tqdm(range(1, n_yr)):
        t_end = years[i]
        x0 = mig[i].reshape(-1)
        bnds = [list(lu) for lu in zip(x0 * 0.5, x0 * 1.5)]
        for bnd in bnds:
            bnd.sort()

        def fn(x):
            x = x.reshape((101, 2))
            stacked_mig['MigR'][i] = x
            demo_mig = Demography(stacked_mig)
            model_mig = ModelAgeSex(demo_mig)
            sol = solve_ivp(model_mig, y0=y0.reshape(-1), t_span=[t_start, t_end], dense_output=True)
            return ((sol.sol(t_end) / demo_mig(t_end)['N'].reshape(-1) - 1) ** 2).sum()

        opt = minimize(fn, x0, bounds=bnds)
        stacked_mig['MigR'][i] = opt.x.reshape((101, 2))

    return Demography(stacked_mig), stacked_mig
