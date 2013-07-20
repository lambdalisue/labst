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
from itertools import product
from sklearn.mixture import GMM


def fit(X, n_components=5, **kwargs):
    """
    Find best fitting by Gaussian Mixture Model

    Arguments:
        X -- data points
        n_components -- maximum number of mixture components

    Returns:
        A list of GMM instance and BICs of each fitting tried
    """
    # create several GMM instance with different covariance_type
    COVARIANCE_TYPES = ['spherical', 'tied', 'diag', 'full']
    properties = list(product(COVARIANCE_TYPES, range(1, n_components+1)))
    models = np.zeros(len(properties), dtype=object)
    for i, (covariance_type, n_component) in enumerate(properties):
        models[i] = GMM(n_component, covariance_type=covariance_type, **kwargs)
        models[i].fit(X)
    # calculate AIC/BIC
    AIC = np.array([m.aic(X) for m in models])
    BIC = np.array([m.bic(X) for m in models])
    # find best model by AIC
    best_model = models[np.argmin(AIC)]
    return best_model, AIC, BIC
