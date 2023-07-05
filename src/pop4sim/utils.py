import numpy as np

__author__ = 'Chu-Chang Ku'
__all__ = ['AgeGroupYMO', 'AgeGroup5Yr', 'gp5yr', 'group_5yr']


AgeGroupYMO = {
    'Y': list(range(15)),
    'M': list(range(15, 65)),
    'O': list(range(65, 101))
}


def gp5yr(n_gp):
    ags, labels = list(), list()

    old = int(5 * (n_gp - 1))

    for i, ag0 in enumerate(list(np.linspace(0, old - 5, n_gp - 1, dtype=int))):
        ags.append(list(range(ag0, ag0 + 5)))
        labels.append(f'{ag0}-{ag0 + 4}')
    ags.append(list(range(old, 101)))
    labels.append(f'{old}+')

    agp = {lab: a for lab, a in zip(labels, ags)}
    return agp


AgeGroup5Yr = gp5yr(n_gp=14)


def group_5yr(x):
    n = np.zeros(21)
    labels = list()
    for i, ag0 in enumerate(list(np.linspace(0, 95, 20, dtype=int))):
        n[i] = x[ag0:(ag0+5)].sum()
        labels.append(f'{ag0}-{ag0+4}')

    n[-1] = x[-1].sum()
    labels.append('100+')
    return n, np.array(labels)


if __name__ == '__main__':
    agp = gp5yr(n_gp=14)
    for k, v in agp.items():
        print(k, ': ', v)

    agp = gp5yr(n_gp=21)
    for k, v in agp.items():
        print(k, ': ', v)
