"""
Regression tools
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_bfgs
from scipy import linalg as la

from . import data

class Regression(object):
    """
    Regression base class
    """
    regCount = 0

    def __init__(self,data):
        self.data = data
        Regression.regCount += 1

    def displayCount(self):
        print "Total instances: %d" % Regression.regCount

    def __del__(self):
        class_name = self.__class__.__name__
        print class_name, "destroyed"

class LinearRegression(Regression):
    """
    Instance of a linear regression analysis.
    """

    def __init__(self,data):
        # init baseclass
        super(LinearRegression,self).__init__(data)

        # set up data
        self.xData = self.data[0] # xData of size (numFeatures,numPoints)
        self.numFeatures = len(self.xData)
        self.numPoints = len(self.xData[0])
        self.yData = self.data[1]

        # Find means and ranges of data for mean Normalization
        # and feature scaling
        self.means = np.mean(self.xData,1).reshape(self.numFeatures,1)
        self.ranges = np.ptp(self.xData,1).reshape(self.numFeatures,1)

        # normalize
        self.xData -= self.means
        self.xData /= self.ranges

        # add bias feature
        self.xData = np.vstack([np.ones(self.numPoints),self.xData])

        # Set up initial theta
        self.theta = np.random.rand(1 + self.numFeatures)

    def plot(self,featureNum,**kwargs):
        plt.scatter(self.xData[featureNum],self.yData,**kwargs)
        x = np.arange(-.5,.5,.01)
        plt.plot(x,x*self.theta[1] + self.theta[0])
        plt.show()

    def getCost(self,theta):
        """
        Calculates cost according to following:

        J(theta) = 1/m*sum(n,(xData*theta - yData)^2)
        """

        self.cost = (1.0/self.numPoints*
                     np.linalg.norm(np.dot(self.xData.T,theta) - self.yData)**2.)

        return self.cost

    def getCostGradient(self,theta):
        """
        Calulates gradient of cost according to following:

        dJ0 = 1/m*sum(m,xData*theta - yData)
        dJj = 1/m*sum(m,(xData*theta - yData)xdataj)
        """

        self.gradJ = np.zeros(1 + self.numFeatures)
        self.e = np.dot(self.xData.T,theta) - self.yData
        self.gradJ[0] = 1.0/self.numPoints*sum(self.e)
        self.gradJ[1:] = 1.0/self.numPoints*np.dot(self.xData[1:],self.e)

        return self.gradJ

    def getTheta(self):
        """Return current value of theta"""
        return self.theta

    def getError(self):
        """
        Return current value of error:

        e = y - Theta.T*X
    
        """
        return self.yData - np.dot(self.xData.T,self.theta)


    def getExactTheta(self):
        """Return optimal theta val as found from eq. 1/(X'X)*(X'y)"""

        return np.dot(la.pinv(np.dot(self.xData,self.xData.T)),
                      np.dot(self.xData,self.yData))


    def getData(self,state='raw'):
        if state == 'raw':
            return (self.xData[1:] * self.ranges + self.means,self.yData)
        elif state == 'normalized':
            return (self.xData[1:],self.yData)

    def getNewVal(self,x):
        """
        If linear regression, x is list of len numFeatures.
        """

        x -= self.means
        x /= self.ranges

        return self.theta[0] + self.theta[1:]*x
        
    def minimize(self):
        self.theta = fmin_bfgs(self.getCost,self.theta,fprime = self.getCostGradient)
        self.getCost(self.theta)

class PolynomialRegression(LinearRegression):

    def __init__(self,data,n,**kwargs):
        # init baseclass
        # n is order of polynomial
        self.polyOrder = n
        
        # can only have 1 x array (for now)
        if len(data) != 2: raise Exception('Too much data for polynomial regression!')

        xData,yData = data
        for i in xrange(2,n+1):
            xData = np.vstack([xData,data[0]**i])

        super(PolynomialRegression,self).__init__((xData,yData))

        plt.scatter(self.xData[1]*self.ranges[0] + self.means[0],self.yData,**kwargs)

    def __getNewVal(self,x):
        """
        x is a single value
        """

        x = np.array([x**(i+1) for i in xrange(self.polyOrder)]).reshape(self.polyOrder,1)
        x -= self.means
        x /= self.ranges

        return self.theta[0] + np.dot(self.theta[1:],x)

    def getNewData(self,xData):
        g = np.vectorize(self.__getNewVal)
        yData = g(xData)
        plt.plot(xData,yData,lw=6)

        return yData
    
if __name__ == '__main__':

    polyOrder = 2

    data = data.genData(10,
                        slopes= np.random.rand(polyOrder),
                        intercept = -1,
                        domains = [(-.4,.4)],
                        fudge = 2.,
                        distribution = 'normal',
                        polyOrder = polyOrder)


    polyReg = PolynomialRegression(data,2)
    polyReg.minimize()
    newData = polyReg.getNewData(np.arange(-.4,.4,.001))
    plt.show()
    #print polyReg.getData('raw')

