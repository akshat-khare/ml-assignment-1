#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import math
import sys
xi = []
# xinp=np.array(xi)
# print(xinp)
with open(sys.argv[1]) as xfile:
    for line in xfile:
        xi1 , xi2 = (float(s) for s in line.split(','))
        xi.append([1, xi1, xi2])
#         print(xi)?
#         np.append(xi,[[float(line)]],axis=0)
# print(xi)
m=len(xi)
xinp = np.zeros((m,3))
# print(xinp)
for x in range(m):
    xinp[x][0] = 1.0
    xinp[x][1] = float(xi[x][1])
    xinp[x][2] = float(xi[x][2])
# print(xinp)
yi=[]
with open(sys.argv[2]) as yifile:
    for line in yifile:
        yi.append(float(line))
yinp = np.zeros((m,1))
for x in range(m):
    yinp[x][0]=yi[x]
# print(yinp)
theta=np.zeros((3,1))
for x in range(3):
    theta[x]=float(0)
def sigmoid(thetaarg, xarg):
    temp = 0.0;
    for x in range(3):
        temp+= thetaarg[x]*xarg[x]
    return (1.0/(1+math.exp(-temp)))
def firstderivativewrtj(j,thetaarg):
    temp=0.0
    for x in range(m):
        temp+= (yinp[x]-sigmoid(thetaarg, xinp[x]))*xinp[x][j]
    return temp
def firstderivative(thetaarg):
    temp = np.zeros((3,1))
    for x in range(3):
        temp[x] = 0.0
    for x in range(3):
        temp[x] = firstderivativewrtj(x,thetaarg)
    return temp
def secondderivativewrtkj(k,j,thetaarg):
    temp = 0.0
    for x in range(m):
        tempsigmoid = sigmoid(thetaarg, xinp[x])
        temp+= (tempsigmoid)*(1-tempsigmoid)*(xinp[x][j])*(xinp[x][k])
    return -temp
def secondderivative(thetaarg):
    temp = np.zeros((3,3))
    for x in range(3):
        for y in range(3):
            temp[x][y] = 0.0
    for x in range(3):
        for y in range(3):
            temp[x][y] = secondderivativewrtkj(x,y, thetaarg)
    return temp
numiter=0
convergencecriteria = pow(10, -7)
while(True):
    numiter+=1
    tempsecond = secondderivative(theta)
    tempsecondinv = np.linalg.inv(tempsecond)
    tempfirst = firstderivative(theta)
    tempchange = np.dot(tempsecondinv, tempfirst)
    theta= theta - tempchange
    temp=0.0
    for x in range(3):
        temp+= abs(tempchange[x])
    if(temp<convergencecriteria):
        break
print(theta)
print(numiter)


# In[33]:


import matplotlib.pyplot as plt
area = np.pi*3
for x in range(m):
    if(abs(yinp[x][0]-1)<0.001):
        plt.scatter(xinp[x][1], xinp[x][2],s=area, c='grey',alpha=0.5, marker='*')
    else:
        plt.scatter(xinp[x][1], xinp[x][2],s=area, c='blue',alpha=0.5, marker='o' )
# print(max(xinp[:][1]))
ximin = min(min(xinp[:][1]),min(xinp[:][2]))
ximax = max(max(xinp[:][1]),max(xinp[:][2]))
xtest = np.linspace(ximin,ximax,100)
ytest =[]
for x in range(len(xtest)):
    ytest.append((-(theta[0]+theta[1]*xtest[x]))/theta[2])
plt.scatter(xtest, ytest,s=area, c='red', alpha=0.5)

# plt.scatter(xi, yi, s=area, c='orange', alpha=0.5)
plt.title('Scatter plot logistic')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:




