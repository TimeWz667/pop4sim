import numpy as np
from pop4sim.demography import Demography
from pop4sim.fetcher import RawData

__author__ = 'Chu-Chang Ku'
__all__ = ['reform_pars_all']


def reform_pars_all(ext: RawData, mig=False, ty='cont'):
    years = ext.Years
    t_start, t_end = ext.YearSpan
    n_yr = len(years)

    stacked = ext.aggregate(sex=False, agp='All')
    if not mig:
        if ty != 'cont':
            demo = Demography(stacked, 'discrete')
        else:
            demo = Demography(stacked, 'cont')
            return demo
    else:
        demo = Demography(stacked, 'cont')

    # Get model-based migration rates
    mig = np.zeros(n_yr)

    for i, t in enumerate(years[1:-1]):
        pars = demo(t)
        dp = np.log(demo(t + 0.5)['N']) - np.log(demo(t - 0.5)['N'])
        mig[i + 1] = dp - pars['r_birth'] + pars['r_death']

    pars = demo(t_end)
    dp = np.log(pars['N']) - np.log(demo(t_end - 0.5)['N'])
    mig[-1] = dp * 2 - pars['r_birth'] + pars['r_death']

    demo.append_mig(mig)
    return demo
