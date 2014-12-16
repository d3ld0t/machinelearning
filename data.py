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

def genData(numPoints,slopes,domains,intercept,fudge,
            distribution = 'uniform',polyOrder = 1): 
    """
    Returns N-D np.ndarray of linear data within parameters 
    modified by a random smudge factor. For linear data, N is 
    minimum dimension of slopes, intercept, and domains.

    numPoints -- number of points to generate

    Args:

        slopes -- rough slopes of each feature; must be same length as polyOrder
            if polyOrder > 1; otherwise len xData
        intercept -- rough y intercept
        domains -- (min,max) tuple of extent of allowable x-values for each feature
        fudge -- float in [0.,1.] specifiying deviation of magnitude float from
            true line where float = 0.

    KWArgs:

        distribution: uniform or normal dist
        polyOrder: if blank, linear from multiple features. otherwise, polynomial degree 
    """
    if distribution == 'normal':
        kwargs = {'distribution':distribution,
                  'loc':np.average(domains),
                  'scale':np.ptp(domains)/2.0}

    elif distribution == 'uniform':
        kwargs = {'distribution':distribution}

    N = (polyOrder
         if polyOrder > 1 
         else min([len(slopes),len(domains)]))

    xData = np.zeros((N,numPoints),dtype = float)
    yData = np.zeros(numPoints,dtype = float)

    for featureNum in xrange(N):
        if polyOrder > 1:
            # only first feature is random, others are powers up to polyOrder
            # domains is only 1 elem long TODO(anuj) make cleaner
            if featureNum == 0:
                xData = randomArray(numPoints,*domains[0],**kwargs)
            yData += xData**(featureNum + 1.)*slopes[featureNum]
        else: 
            xData[featureNum,:] = randomArray(numPoints,*domains[featureNum],**kwargs)
            yData += xData[featureNum,:]*slopes[featureNum]
        
    yData += (randomArray(numPoints,-fudge,fudge,**kwargs) + intercept)

    return (xData,yData)

            

        



