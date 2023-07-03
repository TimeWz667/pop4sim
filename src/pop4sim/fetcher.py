import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import ssl
import urllib3

__author__ = 'Chu-Chang Ku'
__all__ = ['FetcherWPP', 'fetch_wpp']


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
        self.BaseURL = 'https://population.un.org/dataportalapi/api/v1'

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
            f'/data/indicators/{idx}/locations/{gid}/start/{year0}/end/{year1}?pageSize=1000&variant=Median')

        js = response.json()

        dfs = []
        df = pd.json_normalize(js['data'])
        dfs.append(df)

        while js['nextPage'] != None:
            # Redirect to the next page
            target = js['nextPage'].replace(self.BaseURL, '')
            response = self.get(target)
            js = response.json()

            df = pd.json_normalize(js['data'])
            dfs.append(df)

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
        mt[i] = sel.value
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
    dea = fetcher.fetch(idx=69, gid=gid, year0=year0, year1=year1)
    dea = dea[dea.variant == 'Median']
    dea_f = dea[dea.sex == 'Female']
    dea_m = dea[dea.sex == 'Male']

    print('Fetch birth rate')
    bir = fetcher.fetch(idx=55, gid=gid, year0=year0, year1=year1)
    bir = bir[bir.variant == 'Median']

    print('Fetch sex ratio at birth')
    bsr = fetcher.fetch(idx=58, gid=gid, year0=year0, year1=year1)
    bsr = bsr[bsr.variant == 'Median']

    raw = {
        'pop_f': reform_mta(pop_f), 'pop_m': reform_mta(pop_m),
        'dea_f': reform_mta(dea_f), 'dea_m': reform_mta(dea_m),
        'bir': reform_mt(bir), 'bsr': reform_mt(bsr)
    }

    ext = dict()
    ext['N_F_0'], ext['N_F_m'], ext['N_F_1'] = divide_pop(raw['pop_f'][0] * 1e3)
    ext['N_M_0'], ext['N_M_m'], ext['N_M_1'] = divide_pop(raw['pop_m'][0] * 1e3)
    ext['DeaR_F'] = raw['dea_f'][0] / ext['N_F_m'] * 1e3
    ext['DeaR_M'] = raw['dea_m'][0] / ext['N_M_m'] * 1e3

    prop_f = 1 / (1 + raw['bsr'][0])
    prop_m = 1 - prop_f
    ext['BirR_F'] = raw['bir'][0] * prop_f * 1e-3
    ext['BirR_M'] = raw['bir'][0] * prop_m * 1e-3

    ext['Year'], ext['Age'] = raw['dea_f'][1], raw['dea_f'][2]
    return ext


if __name__ == '__main__':
    ext = fetch_wpp(loc='VN', year0=2000, year1=2010)
    print(ext['N_F_m'])
