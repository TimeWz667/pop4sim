import numpy as np
from pop4sim.demography import Demography
from pop4sim.fetcher import RawData

__author__ = 'Chu-Chang Ku'
__all__ = ['reform_pars_sex']


def reform_pars_sex(ext: RawData, mig=False, ty='cont'):
    years = ext.Years
    t_start, t_end = ext.YearSpan
    n_yr = len(years)

    stacked = ext.aggregate(sex=True, agp='All')
    if not mig:
        if ty != 'cont':
            demo = Demography(stacked, 'discrete')
        else:
            demo = Demography(stacked, 'cont')
            return demo
    else:
        demo = Demography(stacked, 'cont')

    # Get model-based migration rates
    mig = np.zeros_like(stacked['N'])

    for i, t in enumerate(years[1:-1]):
        pars = demo(t)
        p1, p0 = demo(t + 0.5)['N'], demo(t - 0.5)['N']
        dp, p = p1 - p0, (p1 + p0) / 2
        mig[i + 1] = dp / p - pars['r_birth'] * p.sum() / p + pars['r_death']

    pars = demo(t_end)
    p1, p0 = pars['N'], demo(t_end - 0.5)['N']
    dp, p = p1 - p0, (p1 + p0) / 2
    mig[-1] = dp / p * 2 - pars['r_birth'] * p.sum() / p + pars['r_death']

    demo.append_mig(mig)
    return demo
