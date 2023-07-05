import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from pop4sim.demography import Demography
from pop4sim.fetcher import RawData
from pop4sim.model.dy import ModelODE

__author__ = 'Chu-Chang Ku'
__all__ = ['reform_pars_agesex']


def reform_pars_agesex(ext: RawData, agp, mig=False, opt_mig=True, ty='cont'):
    mig = False  # todo unlock for more approaches

    years = ext.Years
    t_span = ext.YearSpan
    n_yr = len(years)

    stacked = ext.aggregate(sex=True, agp=agp)
    if not mig:
        if ty != 'cont':
            demo = Demography(stacked, 'discrete')
        else:
            demo = Demography(stacked, 'cont')
            return demo
    else:
        demo = Demography(stacked, 'cont')
    dims = demo(t_span[0])['N'].shape

    model = ModelODE(demo)

    y0 = model.get_y0(t_span[0])
    sol = solve_ivp(model, y0=y0.reshape(-1), t_span=t_span, dense_output=True)

    # Get model-based migration rates
    mig = list()
    for t in years:
        y = sol.sol(t).reshape((101, 2))
        mig.append(demo.calc_mig(t, y))

    mig = np.stack(mig)

    if not opt_mig:
        if ty != 'cont':
            demo = Demography(stacked, 'discrete')
        demo.append_mig(mig)
        return demo

    t_start = t_span[0]

    for i in tqdm(range(1, n_yr)):
        t_end = years[i]
        x0 = mig[i].reshape(-1)
        bnds = [list(lu) for lu in zip(x0 * 0.5, x0 * 1.5)]
        for bnd in bnds:
            bnd.sort()

        def fn(x):
            m = mig.copy()
            m[i] = x.reshape(dims)
            demo.append_mig(m)
            sol = solve_ivp(model, y0=y0.reshape(-1), t_span=[t_start, t_end], dense_output=True)
            return ((sol.sol(t_end) / demo(t_end)['N'].reshape(-1) - 1) ** 2).sum()

        opt = minimize(fn, x0, bounds=bnds)
        mig[i] = opt.x.reshape(dims)

    if ty != 'cont':
        demo = Demography(stacked, 'discrete')
    demo.append_mig(mig)
    return demo
