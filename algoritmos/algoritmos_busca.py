import numpy as np

MAX_IT = 1000
T_SEM_MELHORIA = 300

class BaseAlgoritmo:
    def __init__(self, funcao, dominio, modo='max'):
        self.funcao = funcao
        self.modo = modo
        self.lim_inf = np.array([dominio[0][0], dominio[1][0]], dtype=float)
        self.lim_sup = np.array([dominio[0][1], dominio[1][1]], dtype=float)
        self.scale = self.lim_sup - self.lim_inf
        self.x_best = None
        self.f_best = None
        self.it = 0
        self.max_it = MAX_IT
        self.t_sem_melhoria = T_SEM_MELHORIA
        self.sem_melhoria = 0

    def f(self, x):
        return float(self.funcao(x[0], x[1]))

    def _is_better(self, f_new, f_old):
        return f_new < f_old if self.modo == 'min' else f_new > f_old

class GlobalRandomSearch(BaseAlgoritmo):
    def reset(self, max_it=MAX_IT):
        self.max_it = max_it
        self.x_best = np.random.uniform(self.lim_inf, self.lim_sup)
        self.f_best = self.f(self.x_best)
        self.it = 0
        self.sem_melhoria = 0

    def step(self):
        if self.it >= self.max_it:
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

    def execute(self, max_it=MAX_IT):
        self.reset(max_it=max_it)
        terminou = False
        while not terminou:
            _, _, terminou = self.step()
        return np.copy(self.x_best), float(self.f_best)

class LocalRandomSearch(BaseAlgoritmo):
    def reset(self, sigma=0.05, epsilon=0.05, max_it=MAX_IT):
        self.sigma = sigma * self.scale
        self.epsilon = epsilon * self.scale
        self.max_it = max_it
        self.x_best = np.random.uniform(self.lim_inf, self.lim_sup)
        self.f_best = self.f(self.x_best)
        self.it = 0
        self.sem_melhoria = 0

    def step(self):
        if self.it >= self.max_it or self.sem_melhoria >= self.t_sem_melhoria:
            return self.x_best, self.f_best, True

        y = self.x_best + np.random.normal(0, self.sigma, size=2)
        y = np.clip(y, self.lim_inf, self.lim_sup)

        if np.all(np.abs(y - self.x_best) <= self.epsilon):
            self.sem_melhoria += 1
        else:
            f_y = self.f(y)
            if self._is_better(f_y, self.f_best):
                self.x_best = y
                self.f_best = f_y
                self.sem_melhoria = 0
            else:
                self.sem_melhoria += 1

        self.it += 1
        terminou = (self.it >= self.max_it) or (self.sem_melhoria >= self.t_sem_melhoria)
        return np.copy(self.x_best), float(self.f_best), terminou

    def execute(self, sigma=0.05, epsilon=0.05, max_it=MAX_IT):
        self.reset(sigma=sigma, epsilon=epsilon, max_it=max_it)
        terminou = False
        while not terminou:
            _, _, terminou = self.step()
        return np.copy(self.x_best), float(self.f_best)

class HillClimbing(BaseAlgoritmo):
    def reset(self, epsilon=0.05, max_it=MAX_IT):
        self.epsilon = epsilon * self.scale
        self.max_it = max_it
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
        return np.copy(self.x_best), float(self.f_best), terminou

    def execute(self, epsilon=0.05, max_it=MAX_IT):
        self.reset(epsilon=epsilon, max_it=max_it)
        terminou = False
        while not terminou:
            _, _, terminou = self.step()
        return np.copy(self.x_best), float(self.f_best)
   
class TemperaSimulada:
    def __init__(self, max_it=4000, T=12.0, alpha=0.9995):
        self.max_it = max_it
        self.T = T
        self.alpha = alpha
        self.x_opt = np.random.randint(0, 8, size=8)
        self.f_opt = self.f(self.x_opt)
        self.historico = [self.f_opt]

    def f(self, x):
        ataques = 0
        for i in range(8):
            for j in range(i+1, 8):
                if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):
                    ataques += 1
        return 28 - ataques

    def perturb(self):
        x = self.x_opt.copy()
        r = np.random.rand()
        if r < 0.4:
            col = np.random.randint(0, 8)
            x[col] = np.random.randint(0, 8)
        elif r < 0.8:
            a, b = np.random.choice(8, 2, replace=False)
            x[a], x[b] = x[b], x[a]
        else:
            a, b = np.random.choice(8, 2, replace=False)
            x[a] = np.random.randint(0, 8)
            x[b] = np.random.randint(0, 8)
        return x

    def search(self):
        for _ in range(self.max_it):
            if self.f_opt == 28:
                return self.x_opt, self.f_opt

            cand = self.perturb()
            f_c = self.f(cand)
            delta = f_c - self.f_opt

            if delta > 0 or np.random.rand() < np.exp(delta / self.T):
                self.x_opt = cand
                self.f_opt = f_c

            self.historico.append(self.f_opt)
            self.T *= self.alpha

        return (None, None) if self.f_opt < 28 else (self.x_opt, self.f_opt)

