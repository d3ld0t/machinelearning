#!/usr/bin/env python
"""
Methods to provide data for machine learning algorithms.
"""

import numpy as np

def randomArray(numPoints,lowerBound,upperBound,**kwargs):
    """
    Returns random np.ndarray of random values

    numPoints -- number of points

    Args:
       lowerBound -- lower bound of data
       upperBound -- upper bound of data
    """
    if kwargs['distribution'] == 'uniform':
        return np.random.rand(numPoints)*(upperBound  - lowerBound) + lowerBound
    elif kwargs['distribution'] == 'normal':
        return np.random.normal(loc = kwargs['loc'], scale = kwargs['scale'],size = numPoints)

def genLinearData(numPoints,slopes,domains,intercept,fudge,distribution = 'uniform'): 
    """
    Returns N-D np.ndarray of linear data within parameters 
    modified by a random smudge factor. N is minimum dimension 
    of slopes, intercepts, and domains.

    numPoints -- number of points to generate

    Args:

        slopes -- rough slopes of each feature
        intercept -- rough y intercept
        domains -- (min,max) tuple of extent of allowable x-values for each feature
        fudge -- float in [0.,1.] specifiying deviation of magnitude float from
            true line where float = 0.
    """
    if distribution == 'normal':
        kwargs = {'distribution':distribution,
                  'loc':np.average(domains),
                  'scale':np.ptp(domains)/2.0}

    elif distribution == 'uniform':
        kwargs = {'distribution':distribution}

    N = min([len(slopes),len(domains)])

    xData = np.zeros((N,numPoints),dtype = float)

    for featureNum in xrange(N):
        xData[featureNum,:] = randomArray(numPoints,*domains[featureNum],**kwargs)
        
    fudgedIntercept = randomArray(numPoints,-fudge,fudge,**kwargs) + intercept

    yData = np.dot(xData.T,slopes) + fudgedIntercept

    return (xData,yData)

            

        



