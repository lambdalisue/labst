#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   Alisue (lambdalisue@hashnote.net)
# URL:      http://hashnote.net/
# Date:     2013-07-18
#
# (C) 2013 hashnote.net, Alisue
#
import math
from numpy import histogram as _histogram
from numpy import histogram2d as _histogram2d
from numpy import histogramdd as _histogramdd


def square_root_choice(X):
    """
    Estimate the number of bins from square root of the number of data points

    Arguments:
        X -- data points (iteratable)

    Return:
        intager, the number of bins estimated

    Formula:
        .. math::
            k = \sqrt(n)
    """
    return math.sqrt(len(X))

def sturgens_formula(X):
    """
    Estimate the number of bins from Sturges' formula

    Arguments:
        X -- data points (iteratable)

    Return:
        intager, the number of bins estimated

    Formula:
        .. math::
            k = [\log{2} n + 1]
    """
    k = math.log(len(X), 2) + 1
    k = math.ceil(k)
    return k

def rice_rule(X):
    """
    Estimate the number of bins from Rice Rule (alternative to Sturge's formula)

    Arguments:
        X -- data points (iteratable)

    Return:
        intager, the number of bins estimated

    Formula:
        .. math::
            k = [2n^{\frac{1}{3}}]
    """
    k = math.pow(float(2*len(X)), float(1)/3)
    k = math.ceil(k)
    return k

DEFAULT_BINS_FORMULA = sturgens_formula
"""Default bin estimation function"""

def histogram(a, bins=None, range=None, normed=False, weights=None, density=None):
    """
    Compute the histogram of a set of data.
    It is based on `numpy.histogram` but `bins` automatically estimated from
    samples.

    Arguments:
        Same as `numpy.histogram`

    Returns:
        Same as `numpy.histogram`
    """
    # calculate bins if necessary
    if not bins:
        bins = DEFAULT_BINS_FORMULA
    if callable(bins):
        bins = bins(a)
    return _histogram(a, bins, range, normed, weights, density)

def histogram2d(x, y, bins=None, range=None, normed=False, weights=None):
    """
    Compute the bi-dimensional histogram of two data samples.
    It is based on `numpy.histogram2d` but `bins` automatically estimated from
    samples.

    Arguments:
        Same as `numpy.histogram2d`

    Returns:
        Same as `numpy.histogram2d`
    """
    # calculate bins if necessary
    if not bins:
        bins = DEFAULT_BINS_FORMULA
    if callable(bins):
        bins = (bins(x), bins(y))
    return _histogram2d(x, y, bins, range, weights)

def histogramdd(samples, bins=None, range=None, normed=False, weights=None):
    """
    Compute the multidimensional histogram of some data.
    It is based on `numpy.histogramdd` but `bins` automatically estimated from
    samples.

    Arguments:
        Same as `numpy.histogramdd`

    Returns:
        Same as `numpy.histogramdd`
    """
    # calculate bins if necessary
    if not bins:
        bins = DEFAULT_BINS_FORMULA
    if callable(bins):
        bins = [bins(x) for x in samples]
    return _histogramdd(samples, bins, range, normed, weights)
