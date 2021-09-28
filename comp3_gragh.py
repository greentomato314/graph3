ddef func(x,y,z):
    return x**2+y**3+z**2


def convertXYZ(X,Y,Z):
    return Z+X/2,(3)**(1/2)/2*X

def Gridmake3(dx=0.01):
    x = []
    y = []
    z = []
    for i in np.arange(0,1+dx,dx):
        for j in np.arange(0,1-i+dx,dx):
            k = 1-i-j
            x.append(i)
            y.append(j)
            z.append(k)
    return np.array(x),np.array(y),np.array(z)


import numpy as np
import matplotlib.pyplot as plt

names = ['A','B','C']
title = 'comp3'
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_aspect('equal', 'datalim')
plt.title(title,fontsize=20)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(bottom=False, left=False, right=False, top=False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

h = np.sqrt(3.0)*0.5

#内側目盛
'''
for i in range(1,10):
    ax1.plot([i/20.0, 1.0-i/20.0],[h*i/10.0, h*i/10.0], color='gray', lw=0.5)
    ax1.plot([i/20.0, i/10.0],[h*i/10.0, 0.0], color='gray', lw=0.5)
    ax1.plot([0.5+i/20.0, i/10.0],[h*(1.0-i/10.0), 0.0], color='gray', lw=0.5)
'''

#外周
ax1.plot([0.0, 1.0],[0.0, 0.0], 'k-', lw=2)
ax1.plot([0.0, 0.5],[0.0, h], 'k-', lw=2)
ax1.plot([1.0, 0.5],[0.0, h], 'k-', lw=2)

#頂点のラベル
ax1.text(0.45, h+0.02, names[0], fontsize=16)
ax1.text(-0.1, -0.02, names[1], fontsize=16)
ax1.text(1.03, -0.02, names[2], fontsize=16)

#軸ラベル
for i in range(1,10):
    ax1.text(0.5+(10-i)/20.0, h*(1.0-(10-i)/10.0), '%d0' % i, fontsize=10)
    ax1.text((10-i)/20.0-0.04, h*(10-i)/10.0+0.02, '%d0' % i, fontsize=10, rotation=300)
    ax1.text(i/10.0-0.03, -0.025, '%d0' % i, fontsize=10, rotation=60)

# データ作る
x0,y0,z0 = Gridmake3(dx=0.002)
val = func(x0,y0,z0)
X,Y = convertXYZ(x0,y0,z0)

# ここからプロット
mappable = ax1.scatter(X,Y,c=val,cmap='jet',alpha=0.9)
fig.colorbar(mappable, ax=ax1)

for i in range(1,10):
    ax1.plot([i/20.0, 1.0-i/20.0],[h*i/10.0, h*i/10.0], color='gray', lw=0.5)
    ax1.plot([i/20.0, i/10.0],[h*i/10.0, 0.0], color='gray', lw=0.5)
    ax1.plot([0.5+i/20.0, i/10.0],[h*(1.0-i/10.0), 0.0], color='gray', lw=0.5)

# 最小or最大点
mode = 'min'
if mode == 'max':
    ind = np.argmax(val)
    x = X[ind]
    y = Y[ind]
    ax1.scatter(x,y,c='w')
    ax1.text(x,y,'(x,y,z)=({:.3f},{:.3f},{:.3f})\nmax:{:.3f}'.format(x0[ind],y0[ind],z0[ind],val[ind]),c='b')
if mode == 'min':
    ind = np.argmin(val)
    x = X[ind]
    y = Y[ind]
    ax1.scatter(x,y,c='w')
    ax1.text(x,y,'(x,y,z)=({:.3f},{:.3f},{:.3f})\nmin:{:.3f}'.format(x0[ind],y0[ind],z0[ind],val[ind]),c='y')

plt.show()
