#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   Alisue (lambdalisue@hashnote.net)
# URL:      http://hashnote.net/
# Date:     2013-07-20
#
# (C) 2013 hashnote.net, Alisue
#
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
import labst.utils.textutils as textutils
import labst.statistics.histogram

COLORS = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

def _create_properties(model):
    # zip all properties of model
    properties = zip(model.weights_, model.means_, model._get_covars())
    # the order of each properties is critical so save it as a first property
    properties = enumerate(properties)
    # sort properties by meanu value
    properties = sorted(properties, key=lambda x: np.sum(x[1][1]))
    return properties

def gaussian(X, model, transposition=False, **kwargs):
    """
    Plot gaussian mixture model by `X` and `model`

    Arguments:
        X -- data points
        model -- Gaussian Mixture Model instance
        transposition -- If true, the graph will be rotate 90 degree in
                         clockwise
        **kwargs -- kwargs of `matplotlib.pyplot.plot`
    """
    # create linear points from samples
    x = np.linspace(np.min(X, axis=0), np.max(X, axis=0), len(X))

    # calculate probability of each points
    logprob, responsibilities = model.eval(x)
    pdf = np.exp(logprob)
    print pdf
    pdf_individual = responsibilities * pdf[:,np.newaxis]

    # plot mixture fitting curve
    if not transposition:
        pl.plot(x, pdf, 'r-', **kwargs)
    else:
        pl.plot(pdf, x, 'r-', **kwargs)

    # remove 'label' from kwargs if exists
    if 'label' in kwargs:
        kwargs.pop('label')

    # plot individual fitting curve
    for i, (weight, mean, covar) in _create_properties(model):
        var = np.diag(covar)[0]
        # create formula of normal deviation
        formula_label = textutils.gaussian_formula(len(X), mean, var)
        formula_label = "%.1f%%: %s" % (weight * 100, formula_label)
        # plot
        pdf = pdf_individual[:,i]
        if not transposition:
            pl.plot(x, pdf, 'k--', label=formula_label, **kwargs)
        else:
            pl.plot(pdf, x, 'k--', label=formula_label, **kwargs)


def gaussian_scatter(X, model, transposition=False, colors=COLORS, **kwargs):
    """
    Plot colored scatter by gaussian mixture model components class

    Arguments:
        X -- data points
        model -- Gaussian Mixture Model instance
        transposition -- If true, the graph will be rotate 90 degree in
                         clockwise
        colors -- an iteratable instance of color strings
        **kwargs -- kwargs of `matplotlib.pyplot.plot`
    """
    # add index number to each data points
    if not transposition:
        X_ = np.c_[np.array(range(1, len(X)+1)), X]
    else:
        X_ = np.c_[X, np.array(range(1, len(X)+1))]

    # create prediction (classification) index of each data points
    Y_ = model.predict(X)

    # create properties and zip properties with colors
    properties = _create_properties(model)
    properties = zip(properties, colors)

    for (i, (weight, mean, covar)), color in properties:
        # create formula of normal deviation
        var = np.diag(covar)[0]
        formula_label = textutils.gaussian_formula(len(X), mean, var)
        formula_label = "%.1f%%: %s" % (weight * 100, formula_label)
        # plot related points
        pl.scatter(X_[Y_ == i, 0], X_[Y_ == i, 1],
            marker='x', s=64,
            color=color, label=formula_label, **kwargs)


def gaussian2d(X, Y, model, colors=COLORS, **kwargs):
    """
    Plot 2D histogram colored scatter by `X`, `Y`, `model`

    Arguments:
        X -- data points of X
        Y -- data points of Y
        model -- Gaussian Mixture Model instance
        transposition -- If true, the graph will be rotate 90 degree in
                         clockwise
        colors -- an iteratable instance of color strings
        **kwargs -- kwargs of `matplotlib.pyplot.plot`
    """
    hist, xedges, yedges = labst.statistics.histogram.histogram2d(X, Y)
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    rdn = lambda: (1-(-1))*np.random.random() + -1
    points = []
    for (i, j), v in np.ndenumerate(hist):
        for m in range(int(v)):
            x = xedges[i] + rdn() * dx/6
            y = yedges[j] + rdn() * dy/6
            points.append((x, y))
    points = np.array(points)
    ax = pl.gca()

    # create prediction (classification) index of each data points
    Y_ = model.predict(points)

    # create properties and zip properties with colors
    properties = _create_properties(model)
    properties = zip(properties, colors)
    for (i, (weight, mean, covar)), color in properties:
        # plot related points
        pl.scatter(points[Y_ == i, 0], points[Y_ == i, 1],
                marker='x', s=64,
                color=color, label="Class %d" % i, **kwargs)
        # plot an ellipse to show the component
        w, v = np.linalg.eigh(covar)
        u = v[0] / np.linalg.norm(v[0])
        w = w / 2
        a = np.arctan2(u[1], u[0])
        a = 180.0 * a / np.pi
        e = mpl.patches.Ellipse(mean, w[0], w[1], 180 + a, color=color)
        e.set_clip_box(ax)
        e.set_alpha(0.3)
        ax.add_artist(e)


if __name__ == '__main__':
    import labst.plot.histogram
    import labst.clustering.gaussian
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
    pl.subplot(131)
    k = labst.statistics.histogram.rice_rule(X)
    hist, bins = labst.statistics.histogram.histogram(X, bins=k)
    labst.plot.histogram.histogram(hist, bins, color='black', alpha=0.7)
    pl.xlabel('X')
    pl.ylabel('Frequency')
    pl.grid()
    # Gaussian Mixture Model fitting
    model, AIC, BIC = labst.clustering.gaussian.fit(X, n_components=5)
    pl.twinx()
    gaussian(X, model, alpha=0.5)
    pl.ylabel('Probability')
    pl.legend()
    # 1D Histogram (transposition)
    pl.subplot(132)
    k = labst.statistics.histogram.rice_rule(X)
    hist, bins = labst.statistics.histogram.histogram(X, bins=k)
    labst.plot.histogram.histogram(hist, bins, transposition=True, color='black', alpha=0.7)
    pl.ylabel('X')
    pl.xlabel('Frequency')
    pl.grid()
    # Gaussian Mixture Model fitting (transposition)
    model, AIC, BIC = labst.clustering.gaussian.fit(X, n_components=5)
    pl.twiny()
    gaussian(X, model, transposition=True, alpha=0.5)
    pl.xlabel('Probability')
    pl.legend()
    # reverse Y axis
    ymin, ymax = pl.ylim()
    pl.ylim(ymax, ymin)
    # Scatter with Gaussian Mixture Model
    pl.subplot(133)
    gaussian_scatter(X, model)
    pl.xlabel('Point No.')
    pl.ylabel('X')
    pl.ylim(ymax, ymin)
    pl.legend()
    pl.grid()


    # 2D Histogram Scatter with Gaussian Mixture Model components
    pl.figure()
    model, AIC, BIC = labst.clustering.gaussian.fit(np.c_[X, Y], n_components=5)
    gaussian2d(X, Y, model)
    pl.legend()
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.grid()

    pl.show()
