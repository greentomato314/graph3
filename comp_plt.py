import numpy as np
import matplotlib.pyplot as plt

def convertXY(x,y):
    X = np.array(x)
    Y = np.array(y)
    return Y

def convertXYZ(x,y,z):
    X = np.array(x)
    Y = np.array(y)
    Z = np.array(z)
    return Z+X/2,np.sqrt(3)/2*X

def convertXYZT(X,Y,Z,T):
    r3 = np.sqrt(3)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    T = np.array(T)
    return (3*Y-3*X)/4,(2*r3*Z-r3*X-r3*Y)/4,T*np.sqrt(6)/3

def Gridmake2(n):
    x0,y0 = [],[]
    for i in np.arange(0,1+n):
        x0.append(i)
        y0.append(n-i)
    return np.array(x0)/n,np.array(y0)/n

def Gridmake3(n):
    x0,y0,z0 = [],[],[]
    for i in np.arange(0,1+n):
        for j in np.arange(0,1-i+n):
            k = n-i-j
            x0.append(i)
            y0.append(j)
            z0.append(k)
    return np.array(x0)/n,np.array(y0)/n,np.array(z0)/n

def Gridmake4(n=10): # n等分
    x,y,z,t = [],[],[],[]
    for i in np.arange(0,1+n):
        for j in np.arange(0,1-i+n):
            for k in np.arange(0,1-j-i+n):
                l = n-i-j-k
                x.append(i)
                y.append(j)
                z.append(k)
                t.append(l)
    return np.array(x)/n,np.array(y)/n,np.array(z)/n,np.array(t)/n

class comp2_plt:
    def __init__(self):
        self.val = []
        self.X = []
        self.names_info = [{'s':'A','fontsize':14,'c':'k'},{'s':'B','fontsize':14,'c':'k'}]
        self.names_pos = [(0,0),(1,0)]
        self.x,self.y = [],[]
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
    
    def set_title(self,title,fontsize=20,c='k',**kwargs):
        self.ax1.set_title(title,**kwargs,fontsize=fontsize,c=c)
    
    def set_names(self,names=None,**kwargs):
        if names != None:
            for i in range(2):
                self.names_info[i]['s'] = names[i]
        for key,val in kwargs.items():
            for i in range(2):
                self.names_info[i][key] = val
    
    def set_name(self,pos,name=None,**kwargs):
        if name!=None:
            self.names_info[pos]['s'] = name
        for key,val in kwargs.items():
            self.names_info[pos][key] = val
    
    def get_minmax(self,mode='min'):
        '''
        直前にプロットしたもののmin,maxの位置と値を返す
        '''
        if len(self.val)==0:
            return None,None,None
        if mode == 'max':
            ind = np.argmax(self.val[-1])
        else:
            ind = np.argmin(self.val[-1])
        return self.x[-1][ind],self.y[-1][ind],self.val[-1][ind]

    def plot_func(self,func,GridN=100,**kwargs):
        '''
        cを指定すると固定色、指定しないとヒートマップに反映
        '''
        x0,y0 = Gridmake2(GridN)
        val = func(x0,y0)
        addX = convertXY(x0,y0)
        self.ax1.plot(addX,val,**kwargs)
    
    def plot_scatter(self,X,Y,val,**kwargs):
        '''
        X+Y+Z = 1 にしてください。バグります。
        cを指定すると固定色、指定しないとヒートマップに反映
        '''
        addX = convertXY(X,Y)
        self.ax1.scatter(addX,val,**kwargs)
        
    def plot_text(self,x,y,text,**kwargs):
        self.ax1.text(x,y,text,**kwargs)
    
    def show(self):
        self.ax1.set_xlim([0,1])
        ymin,ymax = self.ax1.get_ylim()
        self.ax1.text(self.names_pos[0][0],self.names_pos[0][1]+ymax+(ymax-ymin)*0.05,**self.names_info[0],horizontalalignment='center')
        self.ax1.text(self.names_pos[1][0],self.names_pos[1][1]+ymax+(ymax-ymin)*0.05,**self.names_info[1],horizontalalignment='center')
        self.ax1.legend()
        plt.show()

class comp3_plt:
    def __init__(self):
        self.val = []
        self.X,self.Y = [],[]
        self.names_info = [{'s':'A','fontsize':14,'c':'k'},{'s':'B','fontsize':14,'c':'k'},{'s':'C','fontsize':14,'c':'k'}]
        self.names_pos = [(0,0),(0,0),(0,0)]
        self.x,self.y,self.z = [],[],[]
        self.plot_info = []
        self.reflected = True
        self.cbar = {'cmap':'jet','vmax':-10**9,'vmin':10**9}
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_aspect('equal', 'datalim')
    
    def set_title(self,title,fontsize=20,c='k',**kwargs):
        self.ax1.set_title(title,**kwargs,fontsize=fontsize,c=c)
    
    def set_names(self,names=None,**kwargs):
        if names != None:
            for i in range(3):
                self.names_info[i]['s'] = names[i]
        for key,val in kwargs.items():
            for i in range(3):
                self.names_info[i][key] = val
    
    def set_name(self,pos,name=None,**kwargs):
        if name!=None:
            self.names_info[pos]['s'] = name
        for key,val in kwargs.items():
            self.names_info[pos][key] = val
    
    def set_colorbar(self,**kwargs):
        for key,val in kwargs.items():
            self.cbar[key] = val
    
    def get_minmax(self,mode='min'):
        '''
        直前にプロットしたもののmin,maxの位置と値を返す
        '''
        if len(self.val)==0:
            return None,None,None,None
        if mode == 'max':
            ind = np.argmax(self.val[-1])
        else:
            ind = np.argmin(self.val[-1])
        return self.x[-1][ind],self.y[-1][ind],self.z[-1][ind],self.val[-1][ind]


    def reflect(self):
        '''
        プロットを反映
        '''
        bar = False
        for i in range(len(self.val)):
            if 'c' in self.plot_info[i]:
                self.ax1.scatter(self.X[i],self.Y[i],**self.plot_info[i])
            else:
                self.mappable = self.ax1.scatter(self.X[i],self.Y[i],**self.plot_info[i],**self.cbar,c=self.val[i])
                bar = True
        if bar: self.fig.colorbar(self.mappable, ax=self.ax1)
        self.val = []
        self.X,self.Y = [],[]
        self.x = []
        self.y = []
        self.z = []
        self.plot_info = []
        self.cbar = {'cmap':'jet','vmax':-10**9,'vmin':10**9}
        self.reflected = True
    
    def plot_func(self,func,GridN=100,**kwargs):
        '''
        cを指定すると固定色、指定しないとヒートマップに反映
        '''
        x0,y0,z0 = Gridmake3(GridN)
        self.x.append(x0)
        self.y.append(y0)
        self.z.append(z0)
        self.val.append(func(x0,y0,z0))
        addX,addY = convertXYZ(x0,y0,z0)
        self.X.append(addX)
        self.Y.append(addY)
        self.plot_info.append(kwargs)
        if not ('c' in kwargs):
            self.cbar['vmax'] = max(self.cbar['vmax'],np.max(self.val[-1]))
            self.cbar['vmin'] = min(self.cbar['vmin'],np.min(self.val[-1]))
        self.reflected = False
    
    def plot_scatter(self,X,Y,Z,val,**kwargs):
        '''
        X+Y+Z = 1 にしてください。バグります。
        cを指定すると固定色、指定しないとヒートマップに反映
        '''
        self.x.append(np.array(X))
        self.y.append(np.array(Y))
        self.z.append(np.array(Z))
        self.val.append(np.array(val))
        addX,addY = convertXYZ(X,Y,Z)
        self.X.append(addX)
        self.Y.append(addY)
        self.plot_info.append(kwargs)
        if not ('c' in kwargs):
            self.cbar['vmax'] = max(self.cbar['vmax'],np.max(self.val[-1]))
            self.cbar['vmin'] = min(self.cbar['vmin'],np.min(self.val[-1]))
        self.reflected = False
    
    def plot_text(self,x,y,z,text,**kwargs):
        X,Y = convertXYZ(x,y,z)
        self.ax1.text(X,Y,text,**kwargs)
    
    def show(self):
        if self.reflected == False:
            self.reflect()

        h = np.sqrt(3.0)*0.5
        # 頂点のラベル
        self.ax1.text(0.5+self.names_pos[0][0], h+0.02+self.names_pos[0][1],**self.names_info[0],horizontalalignment='center')
        self.ax1.text(0+self.names_pos[1][0], -0.07+self.names_pos[1][1],**self.names_info[1],horizontalalignment='center')
        self.ax1.text(1+self.names_pos[2][0], -0.07+self.names_pos[2][1],**self.names_info[2],horizontalalignment='center')
        # 外側の太線
        self.ax1.plot([0.0, 1.0],[0.0, 0.0], 'k-', lw=2)
        self.ax1.plot([0.0, 0.5],[0.0, h], 'k-', lw=2)
        self.ax1.plot([1.0, 0.5],[0.0, h], 'k-', lw=2)
        # 軸ラベル
        for i in range(1,10):
            self.ax1.text(0.5+(10-i)/20.0, h*(1.0-(10-i)/10.0), '%d0' % i, fontsize=10)
            self.ax1.text((10-i)/20.0-0.04, h*(10-i)/10.0+0.02, '%d0' % i, fontsize=10, rotation=300)
            self.ax1.text(i/10.0-0.03, -0.025, '%d0' % i, fontsize=10, rotation=60)
        # 内側の線
        for i in range(1,10):
            self.ax1.plot([i/20.0, 1.0-i/20.0],[h*i/10.0, h*i/10.0], color='gray', lw=0.5)
            self.ax1.plot([i/20.0, i/10.0],[h*i/10.0, 0.0], color='gray', lw=0.5)
            self.ax1.plot([0.5+i/20.0, i/10.0],[h*(1.0-i/10.0), 0.0], color='gray', lw=0.5)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        self.ax1.legend()
        plt.show()


class comp4_plt:
    def __init__(self):
        self.val = []
        self.X,self.Y,self.Z = [],[],[]
        self.names_info = [{'s':'A','fontsize':14,'c':'k'},{'s':'B','fontsize':14,'c':'k'},{'s':'C','fontsize':14,'c':'k'},{'s':'D','fontsize':14,'c':'k'}]
        self.names_pos = [(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
        self.x,self.y,self.z,self.t = [],[],[],[]
        self.plot_info = []
        self.reflected = True
        self.cbar = {'cmap':'jet','vmax':-10**9,'vmin':10**9}
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(projection='3d')
        self.ax1.axis("off")
    
    def set_title(self,title,fontsize=20,c='k',**kwargs):
        self.ax1.set_title(title,**kwargs,fontsize=fontsize,c=c)
    
    def set_names(self,names=None,**kwargs):
        if names != None:
            for i in range(4):
                self.names_info[i]['s'] = names[i]
        for key,val in kwargs.items():
            for i in range(4):
                self.names_info[i][key] = val
    
    def set_name(self,pos,name=None,**kwargs):
        if name!=None:
            self.names_info[pos]['s'] = name
        for key,val in kwargs.items():
            self.names_info[pos][key] = val
    
    def set_colorbar(self,**kwargs):
        for key,val in kwargs.items():
            self.cbar[key] = val
    
    def get_minmax(self,mode='min'):
        '''
        直前にプロットしたもののmin,maxの位置と値を返す
        '''
        if len(self.val)==0:
            return None,None,None,None,None
        if mode == 'max':
            ind = np.argmax(self.val[-1])
        else:
            ind = np.argmin(self.val[-1])
        return self.x[-1][ind],self.y[-1][ind],self.z[-1][ind],self.t[-1][ind],self.val[-1][ind]


    def reflect(self):
        '''
        プロットを反映
        '''
        bar = False
        for i in range(len(self.val)):
            if 'c' in self.plot_info[i]:
                self.ax1.scatter(self.X[i],self.Y[i],self.Z[i],**self.plot_info[i])
            else:
                self.mappable = self.ax1.scatter(self.X[i],self.Y[i],self.Z[i],**self.plot_info[i],**self.cbar,c=self.val[i])
                bar = True
        if bar: self.fig.colorbar(self.mappable, ax=self.ax1)
        self.val = []
        self.X,self.Y,self.Z = [],[],[]
        self.x,self.y,self.z,self.t = [],[],[],[]
        self.plot_info = []
        self.cbar = {'cmap':'jet','vmax':-10**9,'vmin':10**9}
        self.reflected = True
    
    def plot_func(self,func,GridN=10,**kwargs):
        '''
        cを指定すると固定色、指定しないとヒートマップに反映
        '''
        x0,y0,z0,t0 = Gridmake4(GridN)
        self.x.append(x0)
        self.y.append(y0)
        self.z.append(z0)
        self.t.append(t0)
        self.val.append(func(x0,y0,z0,t0))
        addX,addY,addZ = convertXYZT(x0,y0,z0,t0)
        self.X.append(addX)
        self.Y.append(addY)
        self.Z.append(addZ)
        self.plot_info.append(kwargs)
        if not ('c' in kwargs):
            self.cbar['vmax'] = max(self.cbar['vmax'],np.max(self.val[-1]))
            self.cbar['vmin'] = min(self.cbar['vmin'],np.min(self.val[-1]))
        self.reflected = False
    
    def plot_scatter(self,X,Y,Z,T,val,**kwargs):
        '''
        X+Y+Z+T = 1 にしてください。バグります。
        cを指定すると固定色、指定しないとヒートマップに反映
        '''
        self.x.append(np.array(X))
        self.y.append(np.array(Y))
        self.z.append(np.array(Z))
        self.z.append(np.array(T))
        self.val.append(np.array(val))
        addX,addY,addZ = convertXYZT(X,Y,Z,T)
        self.X.append(addX)
        self.Y.append(addY)
        self.Z.append(addZ)
        self.plot_info.append(kwargs)
        if not ('c' in kwargs):
            self.cbar['vmax'] = max(self.cbar['vmax'],np.max(self.val[-1]))
            self.cbar['vmin'] = min(self.cbar['vmin'],np.min(self.val[-1]))
        self.reflected = False
    
    def plot_text(self,x,y,z,t,text,**kwargs):
        X,Y,Z = convertXYZT(x,y,z,t)
        self.ax1.text(X,Y,Z,text,**kwargs)
    
    def show(self):
        if self.reflected == False:
            self.reflect()

        h = np.sqrt(3.0)*0.5
        # 頂点座標
        x1 = tuple(convertXYZT(1,0,0,0))
        y1 = tuple(convertXYZT(0,1,0,0))
        z1 = tuple(convertXYZT(0,0,1,0))
        t1 = tuple(convertXYZT(0,0,0,1))
        # 頂点のラベル
        r = 1.2
        self.ax1.text(x1[0]*r,x1[1]*r,x1[2]*r, **self.names_info[0],horizontalalignment='center')
        self.ax1.text(y1[0]*r,y1[1]*r,y1[2]*r, **self.names_info[1],horizontalalignment='center')
        self.ax1.text(z1[0]*r,z1[1]*r,z1[2]*r, **self.names_info[2],horizontalalignment='center')
        self.ax1.text(t1[0]*r,t1[1]*r,t1[2]*r, **self.names_info[3],horizontalalignment='center')
        # 外側の太線
        self.ax1.plot([x1[0],y1[0]],[x1[1], y1[1]],[x1[2],y1[2]], 'k-', lw=2)
        self.ax1.plot([x1[0],z1[0]],[x1[1], z1[1]],[x1[2],z1[2]], 'k-', lw=2)
        self.ax1.plot([z1[0],y1[0]],[z1[1], y1[1]],[z1[2],y1[2]], 'k-', lw=2)
        self.ax1.plot([x1[0],t1[0]],[x1[1], t1[1]],[x1[2],t1[2]], 'k-', lw=2)
        self.ax1.plot([t1[0],y1[0]],[t1[1], y1[1]],[t1[2],y1[2]], 'k-', lw=2)
        self.ax1.plot([t1[0],z1[0]],[t1[1], z1[1]],[t1[2],z1[2]], 'k-', lw=2)
        self.ax1.legend()
        plt.show()


if __name__=='__main__':
    # 3元系プロット
    Fig = comp3_plt()
    
    # 適当な関数のヒートマップを図示
    def func(x,y,z):
        return x**2+y**3+z**2
    Fig.plot_func(func,label='func',s=3)
    a,b,c,val = Fig.get_minmax(mode='min')
    Fig.plot_scatter([a],[b],[c],val,c='w')
    Fig.plot_text(a,b,c,'min:{:.3f}\n({:.3f},{:.3f},{:.3f})'.format(val,a,b,c),c='k',horizontalalignment='center')
    Fig.set_colorbar(cmap='autumn')
    Fig.set_names(names=['XXXXX','YYYYY','ZZZZZ'],c='r')
    Fig.set_title('TEST',fontsize=20,c='b')
    Fig.show()


    # 予測と合わせて図示
    import pandas as pd
    from sklearn.ensemble import ExtraTreesRegressor

    Fig = comp3_plt()
    df = pd.read_csv('grass.csv')
    df2 = df.dropna(axis=0)
    X = df2.drop('G/C',axis=1)/100
    y = df2['G/C']
    model = ExtraTreesRegressor()
    model.fit(X,y)
    def func(x,y,z):
        dfp = pd.DataFrame()
        dfp['A'] = x
        dfp['B'] = y
        dfp['C'] = z
        return model.predict(dfp)
    Fig.set_title('G/C')
    Fig.plot_func(func,label='predict',s=3,alpha=0.7)
    #Fig.names_pos[0] = (0.05,0.05) #頂点ラベルの位置をずらすこともできる
    Fig.plot_scatter(df['A']/100,df['B']/100,df['C']/100,df['G/C'],label='exp',marker='^')
    Fig.show()

    # ベイズと組み合わせて実験計画法的な使い方
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel, Matern
    
    kernel = ConstantKernel()*RBF() + WhiteKernel() + ConstantKernel() * DotProduct()
    GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    GP.fit(X, y)

    # 予測値プロット
    Fig = comp3_plt()
    x0,y0,z0 = Gridmake3(100)
    dfG = pd.DataFrame({'A':x0,'B':y0,'C':z0})
    G, SD = GP.predict(dfG, return_std=True)
    Fig.set_title('Bayes')
    Fig.plot_scatter(x0,y0,z0,G,label='predict',s=3,alpha=0.7)
    Fig.plot_scatter(df['A']/100,df['B']/100,df['C']/100,df['G/C'],label='exp',marker='^')
    Fig.set_colorbar(vmin=0,vmax=1)
    Fig.show()

    # 分散値プロット
    Fig = comp3_plt()
    Fig.set_title('Std')
    Fig.plot_scatter(x0,y0,z0,SD,label='std',s=3)
    a,b,c,val = Fig.get_minmax(mode='max')
    Fig.plot_scatter([a],[b],[c],val,c='w')
    Fig.plot_text(a,b,c,'max:{:.3f}\n({:.3f},{:.3f},{:.3f})'.format(val,a,b,c),c='k')
    Fig.set_colorbar(cmap='Reds')
    Fig.show()

    # 以下おまけ
    # 4元系プロット
    Fig = comp4_plt()
    def func(x,y,z,t):
        return x**2+2*y**2+3*z**2+4*t**2
    Fig.plot_func(func,label='func',GridN=10,alpha=0.3)
    Fig.set_title('TEST')
    a,b,c,d,val = Fig.get_minmax(mode='min')
    Fig.plot_scatter([a],[b],[c],[d],val,c='k',label='min')
    Fig.plot_text(a,b,c,d,'min:{:.3f}\n({:.3f},{:.3f},{:.3f},{:.3f})'.format(val,a,b,c,d),c='k',horizontalalignment='center')
    Fig.show()

    # 2元系プロット
    Fig = comp2_plt()
    def func(x,y):
        return x**2+2*y**2
    def func2(x,y):
        return x**2+y**3
    Fig.plot_func(func,label='func')
    Fig.plot_func(func2,label='func2')
    Fig.plot_scatter([0.3,0,5,0,7],[0.7,0.5,0.3],[0.2,0.4,0.3],label='scatter',c='r')
    Fig.show()