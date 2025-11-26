import numpy as np
import matplotlib.pyplot as plt


class BaseAlgoritmo:
    def __init__(self, funcao, dominio, modo='max'):
        """
        funcao: função f(x1, x2)
        dominio: [(x1_min, x1_max), (x2_min, x2_max)]
        modo: 'max' ou 'min'
        """
        self.funcao = funcao
        self.modo = modo
        self.lim_inf = np.array([dominio[0][0], dominio[1][0]], dtype=float)
        self.lim_sup = np.array([dominio[0][1], dominio[1][1]], dtype=float)

        self.x_best = None
        self.f_best = None
        self.it = 0
        self.max_it = 1000
        self.t_sem_melhoria = 50
        self.sem_melhoria = 0

    def f(self, x):
        return self.funcao(x[0], x[1])

    def _is_better(self, f_new, f_old):
        if self.modo == 'min':
            return f_new < f_old
        else:
            return f_new > f_old

class GlobalRandomSearch(BaseAlgoritmo):
    def reset(self, max_it=1000, t_sem_melhoria=50):
        self.max_it = max_it
        self.t_sem_melhoria = t_sem_melhoria
        self.x_best = np.random.uniform(self.lim_inf, self.lim_sup)
        self.f_best = self.f(self.x_best)
        self.it = 0
        self.sem_melhoria = 0

    def step(self):
        if self.it >= self.max_it or self.sem_melhoria >= self.t_sem_melhoria:
            return self.x_best, self.f_best, True

        y = np.random.uniform(self.lim_inf, self.lim_sup)
        f_y = self.f(y)

        if self._is_better(f_y, self.f_best):
            self.x_best = y
            self.f_best = f_y
            self.sem_melhoria = 0
        else:
            self.sem_melhoria += 1

        self.it += 1
        terminou = (self.it >= self.max_it) or (self.sem_melhoria >= self.t_sem_melhoria)
        return self.x_best, self.f_best, terminou

    def execute(self, max_it=1000, t_sem_melhoria=50):
        self.reset(max_it=max_it, t_sem_melhoria=t_sem_melhoria)
        terminou = False
        while not terminou:
            _, _, terminou = self.step()
        return self.x_best, self.f_best

    def search(self, R=30, max_it=1000, t_sem_melhoria=50):
        resultados = []

        for _ in range(R):
            x, fx = self.execute(max_it=max_it,
                                 t_sem_melhoria=t_sem_melhoria)
            resultados.append(fx)

        valores, contagens = np.unique(np.round(resultados, 4), return_counts=True)
        idx = np.argmax(contagens)
        melhor_frequentista = valores[idx]

        print("Global Random Search - resultados:", resultados)
        print("Global Random Search - valor mais frequente:", melhor_frequentista)

        return melhor_frequentista, resultados

class LocalRandomSearch(BaseAlgoritmo):
    def reset(self, sigma=0.1, epsilon=0.1, max_it=1000, t_sem_melhoria=50):
        self.sigma = sigma
        self.epsilon = epsilon
        self.max_it = max_it
        self.t_sem_melhoria = t_sem_melhoria
        self.x_best = np.random.uniform(self.lim_inf, self.lim_sup)
        self.f_best = self.f(self.x_best)
        self.it = 0
        self.sem_melhoria = 0

    def step(self):
        if self.it >= self.max_it or self.sem_melhoria >= self.t_sem_melhoria:
            return self.x_best, self.f_best, True

        y = self.x_best + np.random.normal(0, self.sigma, size=2)

        if np.any(np.abs(y - self.x_best) > self.epsilon):
            self.sem_melhoria += 1
        else:
            y = np.clip(y, self.lim_inf, self.lim_sup)
            f_y = self.f(y)

            if self._is_better(f_y, self.f_best):
                self.x_best = y
                self.f_best = f_y
                self.sem_melhoria = 0
            else:
                self.sem_melhoria += 1

        self.it += 1
        terminou = (self.it >= self.max_it) or (self.sem_melhoria >= self.t_sem_melhoria)
        return self.x_best, self.f_best, terminou

    def execute(self, sigma=0.1, epsilon=0.1, max_it=1000, t_sem_melhoria=50):
        self.reset(sigma=sigma, epsilon=epsilon,
                   max_it=max_it, t_sem_melhoria=t_sem_melhoria)
        terminou = False
        while not terminou:
            _, _, terminou = self.step()
        return self.x_best, self.f_best

    def search(self, R=30, sigma=0.1, epsilon=0.1, max_it=1000, t_sem_melhoria=50):
        resultados = []

        for _ in range(R):
            x, fx = self.execute(sigma=sigma,
                                 epsilon=epsilon,
                                 max_it=max_it,
                                 t_sem_melhoria=t_sem_melhoria)
            resultados.append(fx)

        valores, contagens = np.unique(np.round(resultados, 4), return_counts=True)
        idx = np.argmax(contagens)
        melhor_frequentista = valores[idx]

        print("Local Random Search - resultados:", resultados)
        print("Local Random Search - valor mais frequente:", melhor_frequentista)

        return melhor_frequentista, resultados

class HillClimbing(BaseAlgoritmo):
    def reset(self, epsilon=0.1, max_it=1000, t_sem_melhoria=50):
        self.epsilon = epsilon
        self.max_it = max_it
        self.t_sem_melhoria = t_sem_melhoria
        self.x_best = self.lim_inf.copy()
        self.f_best = self.f(self.x_best)
        self.it = 0
        self.sem_melhoria = 0

    def step(self):
        if self.it >= self.max_it or self.sem_melhoria >= self.t_sem_melhoria:
            return self.x_best, self.f_best, True
        y = self.x_best + np.random.uniform(-self.epsilon, self.epsilon, size=2)
        y = np.clip(y, self.lim_inf, self.lim_sup)

        f_y = self.f(y)

        if self._is_better(f_y, self.f_best):
            self.x_best = y
            self.f_best = f_y
            self.sem_melhoria = 0
        else:
            self.sem_melhoria += 1

        self.it += 1
        terminou = (self.it >= self.max_it) or (self.sem_melhoria >= self.t_sem_melhoria)
        return self.x_best, self.f_best, terminou

    def execute(self, epsilon=0.1, max_it=1000, t_sem_melhoria=50):
        self.reset(epsilon, max_it, t_sem_melhoria)
        terminou = False
        while not terminou:
            _, _, terminou = self.step()
        return self.x_best, self.f_best

    def search(self, R=30, epsilon=0.1, max_it=1000, t_sem_melhoria=50):
        resultados = []

        for _ in range(R):
            x, fx = self.execute(epsilon=epsilon,
                                 max_it=max_it,
                                 t_sem_melhoria=t_sem_melhoria)
            resultados.append(fx)
        valores, contagens = np.unique(np.round(resultados, 4), return_counts=True)
        idx = np.argmax(contagens)
        melhor_frequentista = valores[idx]

        print("Hill Climbing - resultados:", resultados)
        print("Hill Climbing - valor mais frequente:", melhor_frequentista)

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