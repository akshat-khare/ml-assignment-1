#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import math
import sys
xi = []
# xinp=np.array(xi)
# print(xinp)
with open(sys.argv[1]) as xfile:
    for line in xfile:
        xi1 , xi2 = (float(s) for s in line.split('  '))
        xi.append([xi1, xi2])
#         print(xi)?
#         np.append(xi,[[float(line)]],axis=0)
# print(xi)
m=len(xi)
xinp = np.zeros((m,2))
# print(xinp)
for x in range(m):
    xinp[x][0] = float(xi[x][0])
    xinp[x][1] = float(xi[x][1])
# print(xinp)
yi=[]
with open(sys.argv[2]) as yifile:
    for line in yifile:
#         print(line)
        temp = line.strip()
        if(temp=='Alaska'):
            yi.append(0)
        else:
            yi.append(1)
yinp = np.zeros((m,1))
for x in range(m):
    yinp[x][0]=yi[x]
#find phi
# print(yinp)
def findphi():
    temp =0.0;
    for x in range(m):
        if(yi[x]==1):
            temp+=1.0
    return temp/(float(m))
# print(findphi())
phi = findphi()
def findmu(i):
    tempnum=np.zeros((2,1))
    tempdeno=0.0
    for x in range(m):
        if(yi[x]==i):
            tempnum = tempnum + np.transpose([xinp[x]])       
            tempdeno +=1
#         print(tempnum)
#     print(tempnum)
#     print(tempdeno)
    return tempnum/float(tempdeno)
# print(findmu(0))
mu0 = findmu(0)
mu1 = findmu(1)
# print(mu0)
# print(mu1)
def getfastmu(i):
    if(i==0):
        return mu0
    else:
        return mu1

def findsigmacommon():
    temp = np.zeros((2,2))
    for x in range(m):
        ultratemp = np.transpose([xinp[x]])-getfastmu(yi[x])
        temp = temp + np.dot(ultratemp, np.transpose(ultratemp))
    return temp/float(m)
# print(findsigmacommon())
def findsigma(i):
    tempnum = np.zeros((2,2))
    tempdeno = 0
    for x in range(m):
        if(yi[x]==i):
            ultratemp = np.transpose([xinp[x]]) - getfastmu(i)
            tempnum = tempnum + np.dot(ultratemp, np.transpose(ultratemp))
            tempdeno +=1
    return tempnum/(float(tempdeno))
# print(findsigma(0))
# print(findsigma(1))


# In[49]:


def gdalinear(xarg):
    tempsigmacommoninv = np.linalg.inv(findsigmacommon())
    tempA = 2*np.dot((np.transpose(mu0)-np.transpose(mu1)),tempsigmacommoninv)
    tempB = math.log(phi/(1-phi)) + (np.dot(np.transpose(mu0), np.dot(tempsigmacommoninv, mu0)))[0][0] - (np.dot(np.transpose(mu1), np.dot(tempsigmacommoninv, mu1)))[0][0]
#     print(tempA[0])
#     print(tempB)
    tempy= []
    for x in range(len(xarg)):
        tempy.append((tempB-tempA[0][0]*xarg[x])/(tempA[0][1]))
    return tempy

# print(gdalinear(xtest))


# In[43]:


def gdaquad():
    tempsigma0 = findsigma(0)
    tempsigma1 = findsigma(1)
    tempsigma0inv = np.linalg.inv(tempsigma0)
    tempsigma1inv = np.linalg.inv(tempsigma1)
    tempmu0transpose = np.transpose(mu0)
    tempmu1transpose = np.transpose(mu1)
    tempA = tempsigma1inv - tempsigma0inv
    tempB = 2*(np.dot(tempmu0transpose, tempsigma0inv) - np.dot(tempmu1transpose, tempsigma1inv))
    tempC = (np.dot(tempmu1transpose, np.dot(tempsigma1inv, mu1)))[0][0]
    tempC = tempC - (np.dot(tempmu0transpose, np.dot(tempsigma0inv, mu0)))[0][0]
    tempC = tempC + math.log(((np.linalg.det(tempsigma1))*(1-phi))/((np.linalg.det(tempsigma0))*(phi)))
    return [tempA[0][0], tempA[1][1], (tempA[1][0]+tempA[0][1]), tempB[0][0], tempB[0][1], tempC]
# print(gdaquad())


# In[75]:


import matplotlib.pyplot as plt
area = np.pi*3
for x in range(m):
    if(yi[x]==1):
        plt.scatter(xinp[x][0], xinp[x][1],s=area, c='grey',alpha=0.5, marker='*' )
    else:
        plt.scatter(xinp[x][0], xinp[x][1],s=area, c='blue',alpha=0.5, marker='o' )
# print(max(xinp[:][1]))
ximin = xi[0][0]
ximax = xi[0][0]
for x in range(m):
    for y in range(2):
        if(xinp[x][y]< ximin):
            ximin=xinp[x][y]
        if(xinp[x][y]> ximax):
            ximax=xinp[x][y]
# print(ximin)
# print(ximax)
mode = int(sys.argv[3])
if(mode==0):
    xtest = np.linspace(50,200,100)
    ytest = gdalinear(xtest)
    plt.scatter(xtest, ytest,s=area, c='red', alpha=0.5)
elif(mode==1):
    argd = gdaquad()
    print(argd)
    x = np.linspace(50,200,100)
    y = np.linspace(300, 550, 100)
    x, y = np.meshgrid(x, y)
    a,b,c,d,e,f = argd[0], argd[1], argd[2], argd[3], argd[4], argd[5]


    plt.contour(x, y ,(a*x**2+b*y**2+c*x*y+d*x+e*y+f),[0], colors='red' )

# plt.scatter(xi, yi, s=area, c='orange', alpha=0.5)
plt.title('Scatter plot of data points and boundary')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:




