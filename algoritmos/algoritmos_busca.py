import numpy as np
import matplotlib.pyplot as plt

class GlobalRandomSearch:
    def __init__(self,max_it, points):
        self.max_it = max_it
        self.points = points
        self.qtd = points.shape[0]

        #ótimo inicial:
        self.x_opt = np.random.permutation(self.qtd-1)+1
        self.x_opt = np.concatenate(([0],self.x_opt))
        self.f_opt = self.f(self.x_opt)
        self.historico = [self.f_opt]

        #FIGURA
        self.lines = []
        self.fig = plt.figure(1)
        self.ax = self.fig.subplots()
        self.ax.set_title(r"Global Random Search")
        self.ax.scatter(self.points[:,0],self.points[:,1])
        self.update_plot()

    def clear_lines(self):
        for line in self.lines:
            line.remove()
        self.lines = []

    def update_plot(self):
        for i in range(self.qtd):
            p1 = self.points[self.x_opt[i]]
            p2 = self.points[self.x_opt[(i+1)%self.qtd]]
            if i  == 0:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='m')
            elif i == self.qtd-1:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='b')
            else:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='k')
            self.lines.append(line[0])
            
    def f(self,x):
        d = 0
        for i in range(self.qtd):
            p1 = self.points[x[i]]
            p2 = self.points[x[(i+1)%self.qtd]]
            # d += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            d+= np.linalg.norm(p1-p2)
        return d
    def perturb(self):
        x_cand = np.random.permutation(self.qtd-1)+1
        x_cand = np.concatenate(([0],x_cand))
        return x_cand
    
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)
            self.historico.append(self.f_opt)
            if f_cand < self.f_opt:
                self.x_opt = x_cand
                self.f_opt = f_cand
                plt.pause(.5)
                self.clear_lines()
                self.update_plot()
            it+=1
        plt.figure(2)
        plt.title("Global Random Search")
        plt.plot(self.historico)
        plt.grid()



class LocalRandomSearch:
    def __init__(self,max_it,epsilon, points):
        self.epsilon = epsilon
        self.max_it = max_it
        self.points = points
        self.qtd = points.shape[0]

        #ótimo inicial:
        self.x_opt = np.random.permutation(self.qtd-1)+1
        self.x_opt = np.concatenate(([0],self.x_opt))
        self.f_opt = self.f(self.x_opt)
        self.historico = [self.f_opt]

        #FIGURA
        self.lines = []
        self.fig = plt.figure(3)
        self.ax = self.fig.subplots()
        self.ax.set_title("Local Random Search")
        self.ax.scatter(self.points[:,0],self.points[:,1])
        self.update_plot()

    def clear_lines(self):
        for line in self.lines:
            line.remove()
        self.lines = []

    def update_plot(self):
        for i in range(self.qtd):
            p1 = self.points[self.x_opt[i]]
            p2 = self.points[self.x_opt[(i+1)%self.qtd]]
            if i  == 0:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='m')
            elif i == self.qtd-1:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='b')
            else:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='k')
            self.lines.append(line[0])

    def f(self,x):
        d = 0
        for i in range(self.qtd):
            p1 = self.points[x[i]]
            p2 = self.points[x[(i+1)%self.qtd]]
            # d += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            d+= np.linalg.norm(p1-p2)
        return d
    
    def perturb(self):
        x_cand = np.copy(self.x_opt)
        indexes1 = (np.random.permutation(self.qtd-1)+1)
        indexes1 = indexes1[:self.epsilon]
        indexes2 = np.random.permutation(indexes1)
        x_cand[indexes1] = x_cand[indexes2]
        return x_cand
    
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)
            self.historico.append(self.f_opt)
            if f_cand < self.f_opt:
                self.x_opt = x_cand
                self.f_opt = f_cand
                plt.pause(.5)
                self.clear_lines()
                self.update_plot()
            it+=1
        plt.figure(4)
        plt.plot(self.historico)
        plt.title("Local Random Search")
        plt.grid()

class TemperaSimulada:
  def __init__(self,max_it,epsilon, points, T, sigma):
        self.epsilon = epsilon
        self.max_it = max_it
        self.points = points
        self.qtd = points.shape[0]
        self.T = T
        self.sigma = sigma

        #ótimo inicial:
        self.x_opt = np.random.permutation(self.qtd-1)+1
        self.x_opt = np.concatenate(([0],self.x_opt))
        self.f_opt = self.f(self.x_opt)
        self.historico = [self.f_opt]

  def f(self,x):
        d = 0
        for i in range(self.qtd):
            p1 = self.points[x[i]]
            p2 = self.points[x[(i+1)%self.qtd]]
            # d += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            d+= np.linalg.norm(p1-p2)
        return d
  
  def perturb(self):
        x_cand = np.copy(self.x_opt)
        indexes1 = (np.random.permutation(self.qtd-1)+1)
        indexes1 = indexes1[:self.epsilon]
        indexes2 = np.random.permutation(indexes1)
        x_cand[indexes1] = x_cand[indexes2]
        return x_cand
  
  def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)
            P_ij = np.exp(-((f_cand-self.f_opt))/self.T)

            if f_cand < self.f_opt or P_ij >= np.random.uniform(0,1):
                self.f_opt = f_cand
                self.x_opt = x_cand

            self.historico.append(self.f_opt)
            self.T = self.T*.98
            plt.figure(4)
            plt.plot(self.historico)
            plt.title("Tempera Simulada")
            plt.grid()
            it+=1
