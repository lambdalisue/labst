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

MICRO = u"\u00b5"
SIGMA = u"\u03c3"
PLMIN = u"\u00b1"


def gaussian_formula(n, mu, sigma, confidence_interval=True):
    """
    Create gaussian formula from parameters

    Arguments:
        n -- the number of sample points
        mu -- the mean of the gaussian formula
        sigma -- the variance of the gaussian formula
        confidence_interval -- if true, include 95% confidence interval

    Returns:
        string, a gaussian formula
    """
    sd = math.sqrt(sigma)        # SD = sqrt(variance)
    se = sd / math.sqrt(n)       # SE = SD / sqrt(N)
    f = "%s=%.1f%s%.1e, %s=%.1e" % (MICRO, mu, PLMIN, se, SIGMA, sd)
    # add 95% confidence interval if required
    if confidence_interval:
        l = mu - 1.96*sd    # 95% confidence interval
        r = mu + 1.96*sd
        f = "%s, 95%%=%.1f~%.1f" % (f, l, r)
    return f
