import numpy as np

__author__ = 'Chu-Chang Ku'
__all__ = ['group_5yr']


def group_5yr(x):
    n = np.zeros(21)
    labels = list()
    for i, ag0 in enumerate(list(np.linspace(0, 95, 20, dtype=int))):
        n[i] = x[ag0:(ag0+5)].sum()
        labels.append(f'{ag0}-{ag0+4}')

    n[-1] = x[-1].sum()
    labels.append('100+')
    return n, np.array(labels)
