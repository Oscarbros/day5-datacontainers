# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:51:01 2021

@author: oscbr226
"""

from scipy import linalg as alg
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import ttest_ind
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

'''Linear algebra with scipy '''
#Problem a-d
A = np.array(range(1,10)).reshape([3,3])
b = np.array([1,2,3])

x = alg.solve(A,b)

sanityCheck = np.matmul(A,x)
# print(sanityCheck)

#Problem e
B = np.random.random([3,3])
sol = alg.solve(A,B)
print(sol)

sanC = np.matmul(A,B)
print('Matrix B:\n' + str(B))
print('Sanity check of matrix multiplication:\n' + str(sanC))

#Problem f
vecsEig = alg.eig(A)
valEigs = alg.eigvals(A)
print('Eigenvectors:\n' + str(vecsEig[1]))
print('Eigenvalues:\n' + str(valEigs))

#Problem g
invA = alg.inv(A)
detInvA = alg.det(invA)
print('Inverse determinant of A:\n' + str(detInvA))

#Problem h
orders = range(1,np.shape(A)[0])
for order in orders:
    normA = alg.norm(A,order)
    print('Norm of A with order ' + str(order) + ':\n' + str(normA))
    
    
''' Statistics with scipy '''
##Problem a
mu = np.random.random(1)
xPoiRange = np.arange(poisson.ppf(0.01,mu), poisson.ppf(0.99,mu))

#PMF
fig, ax = plt.subplots(1,1)
ax.plot(xPoiRange, poisson.pmf(xPoiRange,mu), 'bo', ms=8,label='Poisson pmf')
ax.vlines(xPoiRange, 0, poisson.pmf(xPoiRange,mu), colors='r', lw=3, alpha=0.5)
ax.set_title('Probability mass function of poisson distributed variable')
plt.show()

#CDF
fig1,ax1 = plt.subplots(1,1)
poiCdf = poisson.cdf(xPoiRange,mu)
ax1.plot(xPoiRange, poisson.cdf(mu,xPoiRange))
ax1.set_title('Cumulative distribution function of poisson distributed variable')
plt.show()

#Histogram
randPois = poisson.rvs(mu,size=1000)
fig2,ax2 = plt.subplots(1,1)
ax2.set_title('Histogram of 1000 poisson distributed values')
ax2.hist(randPois)
plt.show()

##Problem b

#PDF
normFig1, axNorm1 = plt.subplots(1,1)
normX = np.linspace(norm.ppf(0.01), norm.ppf(0.99),1000)
axNorm1.plot(normX, norm.pdf(normX), color='m')
axNorm1.set_title('Probability density function of a normal distribution')
plt.show()

#CDF
normFig2, axNorm2 = plt.subplots(1,1)
normVals = norm.ppf([0, 0.5, 1])
normCdf = norm.cdf(normVals)
axNorm2.plot(normCdf)
axNorm2.set_title('Cumulative distribution function of a normal distribution')
plt.show()

#Histogram
normNums = norm.rvs(size=1000)
normFig3, axNorm3 = plt.subplots(1,1)
axNorm3.set_title('Histogram of normally distributed values')
axNorm3.hist(normNums, density=True, histtype='stepfilled')
plt.show()

##Problem c
meanRandPois = np.mean(randPois)
meanRandNorm = np.mean(normNums)
distTest = ttest_ind(randPois,normNums)
print('Result from T-test:\n' + str(distTest))










