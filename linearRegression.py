#!/usr/bin/env python
"""
Linear regression tools
"""

import data
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_bfgs,fmin

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
        super(LinearRegression,self).__init__(data)
        self.xData = self.data[0]
        self.numFeatures = len(self.xData)
        self.numPoints = len(self.xData[0])
        self.xData = np.vstack([np.ones(self.numPoints),self.xData])
        self.yData = self.data[1]
        self.theta = np.random.rand(1 + self.numFeatures)

    def plot(self,featureNum,**kwargs):
        plt.scatter(self.xData[featureNum],self.yData,**kwargs)
        plt.scatter(self.xData[featureNum],np.dot(self.xData.T,self.theta),marker='x')
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

    def minimize(self):
        self.theta = fmin(self.getCost,self.theta,xtol=1.e-8)#,fprime=linReg.costGradient)
        self.getCost(self.theta)



data = data.genLinearData(100,
                          slopes= [-12.],
                          intercept = 4.,
                          domains = [(-3.,4.)],
                          fudge = 80.,
                          distribution = 'normal')
def cost(x):
    return sum(x**2)
linReg = LinearRegression(data)
print linReg.theta
linReg.minimize()
print linReg.theta
linReg.plot(1)
