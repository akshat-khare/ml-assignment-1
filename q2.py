#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import sys
xi = []
# xinp=np.array(xi)
# print(xinp)
with open(sys.argv[1]) as xfile:
    for line in xfile:
        xi.append(float(line))
#         print(xi)?
#         np.append(xi,[[float(line)]],axis=0)
# print(xi)
m=len(xi)
xinp = np.zeros((m,2))
# print(xinp)
for x in range(m):
    xinp[x][0] = 1.0
    xinp[x][1] = float(xi[x])
# print(xinp)
yi=[]
with open(sys.argv[2]) as yifile:
    for line in yifile:
        yi.append(float(line))
yinp = np.zeros((m,1))
for x in range(m):
    yinp[x][0]=yi[x]
xtranspose = np.transpose(xinp)
# print(xtranspose)
xtx = np.dot(xtranspose, xinp)
# print(xtx)
xtxinv = np.linalg.inv(xtx)
# print(xtxinv)
xtxinvxt=np.dot(xtxinv, xtranspose)
# print(xtxinvxt)
theta = np.dot(xtxinvxt,yinp)
print(theta)


# In[43]:


import matplotlib.pyplot as plt
z=[]
for i in range(m):
    z.append(float(theta[0]+theta[1]*xi[i]))
plt.plot(xi, z, linestyle='dotted')
area = np.pi*3
plt.scatter(xi, yi, s=area, c='orange', alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[63]:


import math
def lwlr(xarg, tau):
    w = np.zeros((m,m))
#     print(w)
    for x in range(m):
        w[x][x] = math.exp(-((xarg*1.0-xinp[x][1])**2)/(2*((tau*1.0)**2)))
    fxtranspose = np.transpose(xinp)
    fxtw = np.dot(fxtranspose, w)
    fxtwx = np.dot(fxtw,xinp)
    finvpart = np.linalg.inv(fxtwx)
    finvxt = np.dot(finvpart, fxtranspose)
    finvxtw = np.dot(finvxt, w)
    anstheta = np.dot(finvxtw, yinp)
    ans = np.dot([[1,xarg]], anstheta)
    return ans[0][0]
ximin = min(xi)
ximax = max(xi)
xtest = np.linspace(ximin,ximax,100)
ytest = []
testtau = float(sys.argv[3])
for x in range(len(xtest)):
    ytest.append(lwlr(xtest[x], testtau))
area = np.pi*3
plt.scatter(xi, yi, s=area, c='orange', alpha=0.5)
plt.scatter(xtest, ytest, s=area, c='blue', alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
        


# In[ ]:




