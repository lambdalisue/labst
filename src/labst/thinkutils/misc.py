#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   Alisue (lambdalisue@hashnote.net)
# URL:      http://hashnote.net/
# Date:     2013-07-20
#
# (C) 2013 hashnote.net, Alisue
#
import matplotlib.pyplot as pl
import labst.plot.histogram as plot_histogram
import labst.plot.gaussian as plot_gaussian
import labst.clustering.gaussian as clustering_gaussian
import labst.statistics.histogram as statistics_histogram
import labst.utils.textutils as textutils



def classificate(X, bins=None, n_components=5, verbose=True,
        histogram_ylabel='X',
        histogram_xlabel='Frequency',
        gaussian_xlabel='Probability',
        scatter_xlabel='Point No.'):
    # create histogram data
    hist, bins = statistics_histogram.histogram(X, bins=bins)
    # create gaussian mixture model
    model, AIC, BIC = clustering_gaussian.fit(X, n_components)

    # plot histogram
    pl.subplot(121)
    plot_histogram.histogram(hist, bins, transposition=True, color='k', alpha=0.7)
    pl.xlabel(histogram_xlabel)
    pl.ylabel(histogram_ylabel)
    pl.grid()

    # plot gaussian
    pl.twiny()
    plot_gaussian.gaussian(X, model, transposition=True)
    pl.xlabel(gaussian_xlabel)
    pl.legend()

    # reverse y axis
    ymin, ymax = pl.ylim()
    pl.ylim(ymax, ymin)

    # plot classified scatter
    pl.subplot(122)
    plot_gaussian.gaussian_scatter(X, model)
    pl.ylim(ymax, ymin)
    pl.xlabel(scatter_xlabel)
    pl.ylabel(histogram_ylabel)
    pl.minorticks_on()
    pl.legend()
    pl.grid(which='major', alpha=0.5)
    pl.grid(which='minor', alpha=0.2)

    # print result in stdout if verbose
    if verbose:
        print histogram_xlabel
        print "N = %d, Bins = %d" % (len(X), len(bins))
        # individual fitting curve
        properties = zip(model.weights_, model.means_, model._get_covars())
        properties = sorted(properties, key=lambda x: x[1])
        for (weight, mean, covar) in properties:
            var = np.diag(covar)[0]
            # create formula of normal deviation
            formula_label = textutils.gaussian_formula(len(X), mean, var)
            formula_label = "%.1f%%: %s" % (weight * 100, formula_label)
            print formula_label.encode('utf-8')

    # show figure
    pl.show()


if __name__ == '__main__':
    import numpy as np
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

    classificate(X)
