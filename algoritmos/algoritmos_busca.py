import numpy as np
import matplotlib.pyplot as plt

class GlobalRandomSearch:
    def __init__(self, lim_inf=-2, lim_sup=4):
        self.lim_inf = lim_inf
        self.lim_sup = lim_sup

    def f(self, x):
        return np.exp(-(x**2)) + 2*np.exp(-((x-2)**2))

    def execute(self, max_it=1000, t_sem_melhoria=50):

        x_best = np.random.uniform(self.lim_inf, self.lim_sup)
        f_best = self.f(x_best)

        sem_melhoria = 0

        for it in range(max_it):

            y = np.random.uniform(self.lim_inf, self.lim_sup)

            f_y = self.f(y)

            if f_y > f_best:
                x_best = y
                f_best = f_y
                sem_melhoria = 0
            else:
                sem_melhoria += 1

            if sem_melhoria >= t_sem_melhoria:
                break

        return x_best, f_best

    def search(self, R=30, sigma=0.1, max_it=1000, t_sem_melhoria=50):

        resultados = []

        for r in range(R):
            x, fx = self.execute(
                sigma=sigma,
                max_it=max_it,
                t_sem_melhoria=t_sem_melhoria
            )
            resultados.append(fx)

        valores, contagens = np.unique(np.round(resultados, 4), return_counts=True)
        idx = np.argmax(contagens)
        melhor_frequentista = valores[idx]

        print("\nResultados individuais:", resultados)
        print("Resultado frequentista =", melhor_frequentista)

        return melhor_frequentista, resultados

class LocalRandomSearch:
    def __init__(self, lim_inf=-2, lim_sup=4):
        self.lim_inf = lim_inf
        self.lim_sup = lim_sup

    def f(self, x):
        return np.exp(-(x**2)) + 2*np.exp(-((x-2)**2))

    def execute(self, sigma=0.1, epsilon=0.1, max_it=1000, t_sem_melhoria=50):

        x_best = np.random.uniform(self.lim_inf, self.lim_sup)
        f_best = self.f(x_best)

        sem_melhoria = 0

        for it in range(max_it):
            y = x_best + np.random.normal(0, sigma)

            if abs(y - x_best) > epsilon:
                continue

            y = np.clip(y, self.lim_inf, self.lim_sup)

            f_y = self.f(y)

            if f_y > f_best:
                x_best = y
                f_best = f_y
                sem_melhoria = 0
            else:
                sem_melhoria += 1

            if sem_melhoria >= t_sem_melhoria:
                break

        return x_best, f_best

    def search(self, R=30, sigma=0.1, epsilon=0.1, max_it=1000, t_sem_melhoria=50):

        resultados = []

        for r in range(R):
            x, fx = self.execute(
                sigma=sigma,
                epsilon=epsilon,
                max_it=max_it,
                t_sem_melhoria=t_sem_melhoria
            )
            resultados.append(fx)

        valores, contagens = np.unique(np.round(resultados, 4), return_counts=True)
        idx = np.argmax(contagens)
        melhor_frequentista = valores[idx]

        print("Resultados individuais:", resultados)
        print("Melhor resultado frequentista:", melhor_frequentista)

        return melhor_frequentista, resultados


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

class HillClimbing:
    def __init__(self, lim_inf=-2, lim_sup=4):
        self.lim_inf = lim_inf
        self.lim_sup = lim_sup

    def f(self, x):
        return np.exp(-(x**2)) + 2*np.exp(-((x-2)**2))

    def execute(self, epsilon=0.1, max_it=1000, t_sem_melhoria=50):

        x_best = self.lim_inf
        f_best = self.f(x_best)

        sem_melhoria = 0

        for it in range(max_it):
            y = x_best + np.random.uniform(-epsilon, epsilon)
            y = np.clip(y, self.lim_inf, self.lim_sup)

            f_y = self.f(y)

            if f_y > f_best:
                x_best = y
                f_best = f_y
                sem_melhoria = 0
            else:
                sem_melhoria += 1

            if sem_melhoria >= t_sem_melhoria:
                break

        return x_best, f_best

    def search(self, R=30, epsilon=0.1, max_it=1000, t_sem_melhoria=50):

        resultados = []

        for r in range(R):
            x, fx = self.execute(
                epsilon=epsilon,
                max_it=max_it,
                t_sem_melhoria=t_sem_melhoria
            )
            resultados.append(fx)

        valores, contagens = np.unique(np.round(resultados, 4), return_counts=True)
        indice_max = np.argmax(contagens)
        melhor_frequentista = valores[indice_max]

        print("Resultados obtidos em R execuções:", resultados)
        print("Valor mais frequente (frequentista):", melhor_frequentista)

        return melhor_frequentista, resultados

