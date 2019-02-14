#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import math
import sys
# xfile = open(ass1_data/linearX.csv)
xi = []
with open(sys.argv[1]) as xfile:
    for line in xfile:
        xi.append(float(line))
xmean = np.mean(xi)
xvar = np.var(xi)
xvarroot = math.sqrt(xvar)
yi=[]
with open(sys.argv[2]) as yfile:
    for line in yfile:
        yi.append(float(line))
# print(yi)
m=len(xi)
for x in range(m):
    xi[x] = (xi[x]-xmean)/xvarroot
# print(xi)
lr=float(sys.argv[3])
epsilon=pow(10, -12)
theta=[0.0,0.0]
def jtheta(theta):
    temp=float(0)
    for i in range(m):
        temp+=(pow(theta[1]*xi[i]+1.0*theta[0]-yi[i],2)/2.0)
    return temp
# print(jtheta(theta))
numiter=0
theta0list=[0.0]
theta1list=[0.0]
jthetalist=[jtheta(theta)]
while(True):
#     inijtheta = jtheta(theta)
    temp=[0.0,0.0]
    for i in range(m):
        temp[0]+=(yi[i]-theta[0]*1.0-theta[1]*xi[i])*1.0
        temp[1]+=(yi[i]-theta[0]*1.0-theta[1]*xi[i])*xi[i]
    theta[0] = theta[0]+ lr*temp[0]/(float(m))
    theta[1] = theta[1]+ lr*temp[1]/(float(m))
#     finjtheta = jtheta(theta)
    numiter+=1
    # print("number of iterations are "+str(numiter))
    convergingcriteria = lr*(abs(temp[0])+abs(temp[1]))/(float(m))
    # print("converging criteria is "+str(convergingcriteria))
    theta0list.append(theta[0])
    theta1list.append(theta[1])
    jthetalist.append(jtheta(theta))
    if(convergingcriteria<epsilon):
#         print("inijtheta is "+str(inijtheta)+" finjtheta is "+str(finjtheta))
        print("number of iterations are "+str(numiter))
        break
print(theta)


# In[35]:


import matplotlib.pyplot as plt
 
# Create data
# N = 500
# x = np.random.rand(N)
# print(x)
# y = np.random.rand(N)
# colors = (0,0,0)
area = np.pi*3
 
# Plot
z=[]
for i in range(m):
    z.append(float(theta[0]+theta[1]*xi[i]))
plt.plot(xi, z, linestyle='dotted')
plt.scatter(xi, yi, s=area, c='orange', alpha=0.5)
plt.title('Scatter plot and hypothesis function')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[40]:


from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import sys
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# Data for a three-dimensional line
def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines
def f(x, y):
#     return np.sin(np.sqrt(x ** 2 + y ** 2))
    return jtheta([x,y])

x = np.linspace(-1, 3, 25)
y = np.linspace(-2, 2, 25)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
# ax.view_init(60, 35)
fig
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')

timingdiff = float(sys.argv[4])
# print(theta0list)
# print(theta1list)
# print(jthetalist)
line=[np.array([theta0list,theta1list,jthetalist])]
lines = [ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], '.', color='black')[0] for data in line]
line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(line, lines),
                                   interval=float(timingdiff)*1000, blit=False)
plt.show()

plt.ion()
animated_plot = plt.plot(theta0list, theta1list, 'ro')[0]
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Contours of Error Funtion')

q= np.linspace(-1,2, 100)
p= np.linspace(-1,3, 100)
r, s = np.meshgrid(q,p)
j = 0
sum=0
for i in range(m):
    sum+= pow(s*xi[i] + r-yi[i], 2)/2
j=sum/m
CS=plt.contour(r, s,j,[0,0.0001,0.001,0.005,0.01,.02,0.04,0.08,0.16,0.32,0.64,1,2,4,8],colors='k')
plt.clabel(CS, inline=1, fontsize=10)
for i in range(len(theta0list)):
    animated_plot.set_xdata(theta0list[0:i])
    animated_plot.set_ydata(theta1list[0:i])
    # plt.draw()
    plt.pause(float(timingdiff))
plt.show()
# In[4]:
# def jtheta(theta):
#     temp=float(0)
#     for i in range(m):
#         temp+=(pow(theta[1]*xi[i]+1.0*theta[0]-yi[i],2)/2.0)
#     return temp

#normalizing to be done, stop animation remains, trying for different values is remaining

