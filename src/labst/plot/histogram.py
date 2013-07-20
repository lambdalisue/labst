#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   Alisue (lambdalisue@hashnote.net)
# URL:      http://hashnote.net/
# Date:     2013-07-18
#
# (C) 2013 hashnote.net, Alisue
#
import numpy as np
import matplotlib.pyplot as pl


def histogram(hist, bins, transposition=False, **kwargs):
    """
    Plot histogram by `hist`, `bins`

    Arguments:
        hist -- the values of the histogram
        bins -- bin edges
        **kwargs -- kwargs of `matploblib.pyplot.bar` function

    Note:
        `hist` and `bins` are returning values of
        `labst.statistics.histogram.histogram` function
    """
    # calculate width of each bars by alpha
    alpha = 0.7
    width = alpha * (bins[1] - bins[0])
    # calculate the center point of entire histogram
    center = (bins[1:] + bins[:-1]) / 2
    # create new figure
    if not transposition:
        pl.bar(center, hist, align='center', width=width, **kwargs)
    else:
        pl.barh(center, hist, align='center', height=width, **kwargs)


def histogram2d(hist, xedges, yedges, bubbles=False, **kwargs):
    """
    Plot 2D histogram scatter by `hist`, `xedges`, `yedges`

    Arguments:
        hist -- the values of the histogram
        xedges -- x bin edges
        yedges -- y bin edges
        bubbles -- If true, draw bubble instead of points
        **kwargs -- kwargs of `matploblib.pyplot.scatter` function

    Note:
        `hist`, `xedges` and `yedges` are returning values of
        `labst.statistics.histogram.histogram2d` function
    """
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    rdn = lambda: (1-(-1))*np.random.random() + -1
    points = []
    for (i, j), v in np.ndenumerate(hist):
        if bubbles:
            points.append((xedges[i], yedges[j], v))
        else:
            for m in range(int(v)):
                x = xedges[i] + rdn() * dx/6
                y = yedges[j] + rdn() * dy/6
                points.append((x, y))
    points = np.array(points)
    if bubbles:
        sub = pl.scatter(points[:,0], points[:,1],
                marker='o', s=128*points[:,2],
                **kwargs)
    else:
        # set default kwargs
        kwargs['marker'] = kwargs.get('marker', 'x')
        kwargs['s'] = kwargs.get('s', 64)
        sub = pl.scatter(points[:,0], points[:,1], **kwargs)


if __name__ == '__main__':
    import labst.statistics.histogram
    # create sample data
    mu1, sigma1 = 100, 15
    mu2, sigma2 = 30, 5
    mu3, sigma3 = 50, 30
    X = np.r_[np.random.normal(mu1, sigma1, 500),
              np.random.normal(mu2, sigma2, 300),
              np.random.normal(mu3, sigma3, 200)]
    Y = np.r_[np.random.normal(mu1, sigma1, 500),
              np.random.normal(mu2, sigma2, 300),
              np.random.normal(mu3, sigma3, 200)]

    # 1D Histogram
    k = labst.statistics.histogram.rice_rule(X)
    hist, bins = labst.statistics.histogram.histogram(X, bins=k)
    pl.subplot(121)
    histogram(hist, bins, color='black', alpha=0.7)
    pl.xlabel('X')
    pl.ylabel('Frequency')
    pl.grid()
    pl.subplot(122)
    histogram(hist, bins, transposition=True, color='black', alpha=0.7)
    pl.ylabel('X')
    pl.xlabel('Frequency')
    pl.grid()

    # 2D Histogram
    k1 = labst.statistics.histogram.rice_rule(X)
    k2 = labst.statistics.histogram.rice_rule(Y)
    hist, xedges, yedges = labst.statistics.histogram.histogram2d(X, Y, bins=(k1, k2))
    pl.figure()
    pl.subplot(121)
    histogram2d(hist, xedges, yedges, color='black', alpha=0.7)
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.grid()
    pl.subplot(122)
    histogram2d(hist, xedges, yedges, bubbles=True, color='black', alpha=0.7)
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.grid()

    pl.show()
