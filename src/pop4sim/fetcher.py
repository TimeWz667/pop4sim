import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import ssl
import urllib3
from tqdm import tqdm

__author__ = 'Chu-Chang Ku'
__all__ = ['FetcherWPP', 'fetch_wpp', 'RawData']

WPP_BaseURL = 'https://population.un.org/dataportalapi/api/v1'


class RawData:
    def __init__(self, loc, years, pop, dea, bir):
        self.Location = loc
        self.Years = years
        self.YearSpan = years[0], years[-1]
        self.Population = pop
        self.RateDeath = dea
        self.RateBirth = bir

    def aggregate(self, sex=True, agp='All'):
        dimnames = {'Year': [int(yr - 0.5) for yr in self.Years]}

        n0 = np.stack([self.Population['F'], self.Population['M']], 2)
        dr0 = np.stack([self.RateDeath['F'], self.RateDeath['M']], 2)
        br0 = np.stack([self.RateBirth['F'], self.RateBirth['M']], 1)

        nt = n0.sum((1, 2))

        dea0 = dr0 * n0
        bir0 = br0 * n0.sum((1, 2)).reshape((-1, 1))

        if isinstance(agp, dict):
            n_agp = len(agp)
            n_yr = n0.shape[0]

            n1 = np.zeros((n_yr, n_agp, 2))
            dea1 = np.zeros((n_yr, n_agp, 2))
            ageing1 = np.zeros((n_yr, n_agp, 2))

            for i, a in enumerate(agp.values()):
                n1[:, i, 0] = n0[:, a, 0].sum(1)
                n1[:, i, 1] = n0[:, a, 1].sum(1)
                dea1[:, i, 0] = dea0[:, a, 0].sum(1)
                dea1[:, i, 1] = dea0[:, a, 1].sum(1)
                ageing1[:, i] = n0[:, a[-1]]

            ageing1[:, -1, :] = 0
            bir1 = bir0
            dimnames['Age'] = list(agp.keys())
        elif agp == 'All':
            n1 = n0.sum(1)
            dea1 = dea0.sum(1)
            bir1 = bir0
            ageing1 = np.zeros_like(n1)
        else:
            n1 = n0
            dea1 = dea0
            bir1 = bir0
            ageing1 = n0.copy()
            ageing1[:, -1, :] = 0
            dimnames['Age'] = [str(a) for a in range(100)]
            dimnames['Age'].append('100+')

        if not sex:
            n1 = n1.sum(-1)
            dea1 = dea1.sum(-1)
            ageing1 = ageing1.sum(-1)
            bir1 = bir1.sum(-1)
        else:
            dimnames['Sex'] = ['F', 'M']

        dr1 = dea1 / n1
        ar1 = ageing1 / n1
        if sex:
            br1 = bir1 / nt.reshape((-1, 1))
        else:
            br1 = bir1 / nt

        return {
            'dimnames': dimnames,
            'Year': self.Years,
            'N': n1,
            'RateDeath': dr1,
            'RateBirth': br1,
            'RateAgeing': ar1,
            'Deaths': dea1,
            'Births': bir1,
            'Ageings': ageing1
        }

    def __str__(self):
        return f'RawData({self.Location}, from {self.YearSpan[0]} to {self.YearSpan[1]}, agp: Single age)'

    __repr__ = __str__


class DefaultAdaptor(requests.adapters.HTTPAdapter):
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)


class FetcherWPP:
    def __init__(self, adapter=None):
        self.Session = ses = requests.session()
        self.BaseURL = WPP_BaseURL

        if adapter is None:
            ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ctx.options |= 0x4
            adapter = DefaultAdaptor(ctx)
            ses.mount('https://', adapter)

    def get(self, sub_url):
        return self.Session.get(self.BaseURL + sub_url)

    def find_geo(self, loc):
        res = self.get(f'/locations/{loc}')

        if res.status_code == 200:
            return res.json()[0]
        raise ValueError('Unknown locator')

    def fetch(self, gid, idx, year0, year1):
        response = self.get(
            f'/data/indicators/{idx}/locations/{gid}/start/{year0}/end/{year1}')

        js = response.json()

        dfs = []
        df = pd.json_normalize(js['data'])
        dfs.append(df)

        t_end = float(js['data'][-1]['timeLabel'])

        # print(f'--{t_end}')
        with tqdm(total=year1 - year0) as pbar:
            pbar.update(t_end - year0)
            while js['nextPage'] is not None:
                # Redirect to the next page
                target = js['nextPage'].replace(self.BaseURL, '')
                response = self.get(target)
                js = response.json()

                df = pd.json_normalize(js['data'])
                dfs.append(df)

                diff = float(js['data'][-1]['timeLabel']) - t_end
                if diff > 0:
                    pbar.update(diff)
                    t_end += diff
                    # print(f'--{t_end}')

        return pd.concat(dfs)


def reform_mta(df):
    yrs = list(set(df.timeLabel))
    yrs = [int(yr) for yr in yrs]
    yrs.sort()
    ages = list(set(df.ageStart))
    ages.sort()

    mta = np.zeros((len(yrs), len(ages)))

    for i, yr in enumerate(yrs):
        yr = f'{yr}'
        sel = df[df.timeLabel == yr]
        sel = sel.sort_values('ageStart')
        mta[i] = sel.value
    return mta, yrs, ages


def reform_mt(df):
    yrs = list(set(df.timeLabel))
    yrs = [int(yr) for yr in yrs]
    yrs.sort()

    mt = np.zeros(len(yrs))

    for i, yr in enumerate(yrs):
        yr = f'{yr}'
        sel = df[df.timeLabel == yr]
        mt[i] = sel.value.iloc[0]
    return mt, yrs


def divide_pop(ps):
    ps_m = ps[1:-1]
    ps = (ps[:-1] + ps[1:]) / 2
    ps_0, ps_1 = ps[:-1], ps[1:]
    return ps_0, ps_m, ps_1


def fetch_wpp(loc, year0=1970, year1=2030, fetcher=None):
    if fetcher is None:
        fetcher = FetcherWPP()

    geo = fetcher.find_geo(loc)
    gid = geo['id']

    print('Fetch population size')
    pop = fetcher.fetch(idx=47, gid=gid, year0=year0 - 1, year1=year1 + 1)
    pop = pop[pop.variant == 'Median']
    pop_f = pop[pop.sex == 'Female']
    pop_m = pop[pop.sex == 'Male']

    print('Fetch deaths')
    dea = fetcher.fetch(idx=69, gid=gid, year0=year0 - 1, year1=year1 + 1)
    dea = dea[dea.variant == 'Median']
    dea_f = dea[dea.sex == 'Female']
    dea_m = dea[dea.sex == 'Male']

    print('Fetch birth rate')
    bir = fetcher.fetch(idx=55, gid=gid, year0=year0 - 1, year1=year1 + 1)
    bir = bir[bir.variant == 'Median']

    print('Fetch sex ratio at birth')
    bsr = fetcher.fetch(idx=58, gid=gid, year0=year0 - 1, year1=year1 + 1)
    bsr = bsr[bsr.variant == 'Median']

    raw = {
        'pop_f': reform_mta(pop_f), 'pop_m': reform_mta(pop_m),
        'dea_f': reform_mta(dea_f), 'dea_m': reform_mta(dea_m),
        'bir': reform_mt(bir), 'bsr': reform_mt(bsr)
    }

    pop = {
        'F': raw['pop_f'][0],
        'M': raw['pop_m'][0]
    }

    # ext['N_F_0'], ext['N_F_m'], ext['N_F_1'] = divide_pop(raw['pop_f'][0] * 1e3)
    # ext['N_M_0'], ext['N_M_m'], ext['N_M_1'] = divide_pop(raw['pop_m'][0] * 1e3)
    dea = {
        'F': raw['dea_f'][0] / pop['F'],
        'M': raw['dea_m'][0] / pop['M']
    }

    prop_f = 1 / (1 + raw['bsr'][0])
    prop_m = 1 - prop_f

    bir = {
        'F': raw['bir'][0] * prop_f * 1e-3,
        'M': raw['bir'][0] * prop_m * 1e-3
    }

    years = np.array(raw['dea_f'][1]) + 0.5
    return RawData(loc=geo['name'], years=years, pop=pop, dea=dea, bir=bir)


if __name__ == '__main__':
    ext = fetch_wpp(loc='VN', year0=2000, year1=2003)

    print(ext)
    print(ext.Population['F'])
