import numpy as np
import matplotlib.pyplot as plt

class comp3_plt:
    def __init__(self):
        self.val = np.array([])
        self.X,self.Y = np.array([]),np.array([])
        self.names = ['A','B','C']
        self.title = ''
        self.fontsize_title = 20
        self.fontsize_name = 14
        self.x = []
        self.y = []
        self.z = []
        self.a = (0,0)
        self.b = (0,0)
        self.c = (0,0)
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_aspect('equal', 'datalim')

    
    def set_title(self,title=None,fontsize=None):
        if title != None : self.title = title
        if fontsize != None : self.fontsize_title = fontsize
    
    def set_names(self,names=None,fontsize=14,a=(0,0),b=(0,0),c=(0,0)):
        if names != None: self.names=names
        if fontsize != None : self.fontsize_name = fontsize
        if a != None : self.a = a
        if b != None : self.b = b
        if c != None : self.c = c

    def plot_func(self,func,GridN=100,cmap='jet',colorbar=True,alpha=0.9):
        x0,y0,z0 = self.Gridmake3(GridN)
        self.val = func(x0,y0,z0)
        self.X,self.Y = self.convertXYZ(x0,y0,z0)
        self.mappable = self.ax1.scatter(self.X,self.Y,c=self.val,cmap=cmap,alpha=alpha)
        if colorbar: self.fig.colorbar(self.mappable, ax=self.ax1)
    
    def plot_scatter(self,X,Y,Z,val,cmap='jet',colorbar=True,alpha=0.9):
        '''
        X+Y+Z = 1 にしてください。バグります。
        '''
        self.val = val
        self.X,self.Y = self.convertXYZ(X,Y,Z)
        self.mappable = self.ax1.scatter(self.X,self.Y,c=self.val,cmap=cmap,alpha=alpha)
        if colorbar: self.fig.colorbar(self.mappable, ax=self.ax1)

    def Gridmake3(self,n):
        self.x = []
        self.y = []
        self.z = []
        for i in np.arange(0,1+n):
            for j in np.arange(0,1-i+n):
                k = n-i-j
                self.x.append(i)
                self.y.append(j)
                self.z.append(k)
        return np.array(self.x)/n,np.array(self.y)/n,np.array(self.z)/n
    
    def convertXYZ(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        X = np.array(x)
        Y = np.array(y)
        Z = np.array(z)
        return Z+X/2,np.sqrt(3)/2*X
    
    def plot_minmax(self,mode='min',modesize=3,modecolor='w',fontcolor='r',fontsize=10):
        if mode == 'max':
            ind = np.argmax(self.val)
            x = self.X[ind]
            y = self.Y[ind]
            self.ax1.scatter(x,y,c=modecolor,s=modesize)
            self.ax1.text(x,y,'(x,y,z)=({:.3f},{:.3f},{:.3f})\nmax:{:.3f}'.format(self.x[ind],self.y[ind],self.z[ind],self.val[ind]),c=fontcolor,fontsize=fontsize)
        if mode == 'min':
            ind = np.argmin(self.val)
            x = self.X[ind]
            y = self.Y[ind]
            self.ax1.scatter(x,y,c=modecolor,s=modesize)
            self.ax1.text(x,y,'(x,y,z)=({:.3f},{:.3f},{:.3f})\nmin:{:.3f}'.format(self.x[ind],self.y[ind],self.z[ind],self.val[ind]),c=fontcolor,fontsize=fontsize)
    
    def show(self):   
        h = np.sqrt(3.0)*0.5
        self.ax1.plot([0.0, 1.0],[0.0, 0.0], 'k-', lw=2)
        self.ax1.plot([0.0, 0.5],[0.0, h], 'k-', lw=2)
        self.ax1.plot([1.0, 0.5],[0.0, h], 'k-', lw=2)
        #頂点のラベル
        self.ax1.text(0.5+self.a[0], h+0.02+self.a[1], self.names[0], fontsize=self.fontsize_name,horizontalalignment='center')
        self.ax1.text(0+self.b[0], -0.07+self.b[1], self.names[1], fontsize=self.fontsize_name,horizontalalignment='center')
        self.ax1.text(1+self.c[0], -0.07+self.c[1], self.names[2], fontsize=self.fontsize_name,horizontalalignment='center')
        #軸ラベル
        for i in range(1,10):
            self.ax1.text(0.5+(10-i)/20.0, h*(1.0-(10-i)/10.0), '%d0' % i, fontsize=10)
            self.ax1.text((10-i)/20.0-0.04, h*(10-i)/10.0+0.02, '%d0' % i, fontsize=10, rotation=300)
            self.ax1.text(i/10.0-0.03, -0.025, '%d0' % i, fontsize=10, rotation=60)
        plt.title(self.title,fontsize=self.fontsize_title)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.show()


if __name__=='__main__':
    Fig = comp3_plt()
    def func(x,y,z):
        return x**2+y**3+z**2
    Fig.plot_func(func)
    Fig.set_names(names=['XXXXX','YYYYY','ZZZZZ'])
    Fig.plot_minmax(mode='min',modesize=3,modecolor='w',fontsize=7,fontcolor='w')
    Fig.plot_scatter([0.5],[0.25],[0.25],[0.5],colorbar=False,alpha=1)
    Fig.set_title('TEST',fontsize=20)
    Fig.set_names(fontsize=10)
    Fig.show()
