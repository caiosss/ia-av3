import numpy as np
import matplotlib.pyplot as plt

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
        self.max_it = 1000
        self.t_sem_melhoria = 300
        self.sem_melhoria = 0

    def f(self, x):
        return self.funcao(x[0], x[1])

    def _is_better(self, f_new, f_old):
        return f_new < f_old if self.modo == 'min' else f_new > f_old

class GlobalRandomSearch(BaseAlgoritmo):
    def reset(self, max_it=1000):
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

        self.it += 1
        return self.x_best, self.f_best, (self.it >= self.max_it)

    def execute(self, max_it=1000):
        self.reset(max_it=max_it)
        terminou = False
        while not terminou:
            _, _, terminou = self.step()
        return self.x_best, self.f_best

class LocalRandomSearch(BaseAlgoritmo):
    def reset(self, sigma=0.05, epsilon=0.05, max_it=1000):
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

        if np.any(np.abs(y - self.x_best) > self.epsilon):
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
        return self.x_best, self.f_best, terminou

    def execute(self, sigma=0.05, epsilon=0.05, max_it=1000):
        self.reset(sigma=sigma, epsilon=epsilon, max_it=max_it)
        terminou = False
        while not terminou:
            _, _, terminou = self.step()
        return self.x_best, self.f_best

class HillClimbing(BaseAlgoritmo):
    def reset(self, epsilon=0.05, max_it=1000):
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
        return self.x_best, self.f_best, (self.it >= self.max_it or self.sem_melhoria >= self.t_sem_melhoria)

class TemperaSimulada:
    def __init__(self, max_it=4000, T=12.0, alpha=0.9995):
        self.max_it = max_it
        self.T = T
        self.alpha = alpha
        self.qtd = 8
        self.x_opt = np.random.randint(0, 8, size=8)
        self.f_opt = self.f(self.x_opt)
        self.historico = [self.f_opt]

    def f(self, x):
        ataques = 0
        for i in range(8):
            for j in range(i+1, 8):
                if x[i] == x[j]:
                    ataques += 1
                if abs(x[i] - x[j]) == abs(i - j):
                    ataques += 1
        return 28 - ataques

    def perturb(self):
        x_cand = np.copy(self.x_opt)
        r = np.random.rand()
        if r < 0.40:
            col = np.random.randint(0, 8)
            x_cand[col] = np.random.randint(0, 8)
        elif r < 0.80:
            c1, c2 = np.random.choice(8, 2, replace=False)
            x_cand[c1], x_cand[c2] = x_cand[c2], x_cand[c1]
        else:
            cols = np.random.choice(8, 2, replace=False)
            x_cand[cols[0]] = np.random.randint(0, 8)
            x_cand[cols[1]] = np.random.randint(0, 8)
        return x_cand

    def search(self):
        it = 0
        while it < self.max_it and self.f_opt < 28:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)
            delta = f_cand - self.f_opt
            if delta > 0 or np.random.rand() < np.exp(delta / self.T):
                self.x_opt = x_cand
                self.f_opt = f_cand
            self.historico.append(self.f_opt)
            self.T *= self.alpha
            it += 1

        if self.f_opt < 28:
            return None, None
        return self.x_opt, self.f_opt


def f1(x1, x2): return x1**2 + x2**2
dom1 = [(-100, 100), (-100, 100)]
modo1 = 'min'

def f2(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2*np.exp(-((x1-1.7)**2 + (x2-1.7)**2))
dom2 = [(-2, 4), (-2, 5)]
modo2 = 'max'

def f3(x1, x2):
    t1 = -20*np.exp(-0.2*np.sqrt(0.5*(x1**2+x2**2)))
    t2 = -np.exp(0.5*(np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2)))
    return t1 + t2 + 20 + np.e
dom3 = [(-8, 8), (-8, 8)]
modo3 = 'min'

def f4(x1, x2):
    return (x1**2 - 10*np.cos(2*np.pi*x1) + 10) + (x2**2 - 10*np.cos(2*np.pi*x2) + 10)
dom4 = [(-5.12, 5.12), (-5.12, 5.12)]
modo4 = 'min'

def f5(x1, x2):
    t1 = x1*np.cos(x1)/20
    t2 = 2*np.exp(-(x1**2 + (x2-1)**2))
    t3 = 0.01*x1*x2
    return t1 + t2 + t3
dom5 = [(-10,10), (-10,10)]
modo5 = 'max'

def f6(x1, x2):
    return x1*np.sin(4*np.pi*x1) - x2*np.sin(4*np.pi*x2 + np.pi) + 1
dom6 = [(-1,3), (-1,3)]
modo6 = 'max'

PROBLEMAS = [
    ("Problema 1", f1, dom1, modo1),
    ("Problema 2", f2, dom2, modo2),
    ("Problema 3", f3, dom3, modo3),
    ("Problema 4", f4, dom4, modo4),
    ("Problema 5", f5, dom5, modo5),
    ("Problema 6", f6, dom6, modo6),
]

def animar_algoritmos(funcao, dominio, modo, nome, max_frames=150):
    hc = HillClimbing(funcao, dominio, modo)
    lrs = LocalRandomSearch(funcao, dominio, modo)
    grs = GlobalRandomSearch(funcao, dominio, modo)

    hc.reset(epsilon=0.05, max_it=max_frames)
    lrs.reset(sigma=0.05, epsilon=0.05, max_it=max_frames)
    grs.reset(max_it=max_frames)

    hist_hc, hist_lrs, hist_grs = [], [], []
    trail_hc_x, trail_hc_y, trail_hc_z = [], [], []
    trail_lrs_x, trail_lrs_y, trail_lrs_z = [], [], []
    trail_grs_x, trail_grs_y, trail_grs_z = [], [], []

    x1 = np.linspace(dominio[0][0], dominio[0][1], 40)
    x2 = np.linspace(dominio[1][0], dominio[1][1], 40)
    X, Y = np.meshgrid(x1, x2)
    Z = funcao(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(nome)
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.55)

    hc_point = ax.plot([hc.x_best[0]],[hc.x_best[1]],[hc.f_best],'ro')[0]
    lrs_point = ax.plot([lrs.x_best[0]],[lrs.x_best[1]],[lrs.f_best],'g^')[0]
    grs_point = ax.plot([grs.x_best[0]],[grs.x_best[1]],[grs.f_best],'b*')[0]

    line_hc, = ax.plot([], [], [], 'r-', linewidth=1)
    line_lrs, = ax.plot([], [], [], 'g-', linewidth=1)
    line_grs, = ax.plot([], [], [], 'b-', linewidth=1)

    plt.pause(0.2)

    for _ in range(max_frames):
        xh, fh, _ = hc.step()
        hist_hc.append(fh)
        trail_hc_x.append(xh[0]); trail_hc_y.append(xh[1]); trail_hc_z.append(fh)
        hc_point.set_data([xh[0]], [xh[1]])
        hc_point.set_3d_properties([fh])
        line_hc.set_data(trail_hc_x, trail_hc_y)
        line_hc.set_3d_properties(trail_hc_z)

        xl, fl, _ = lrs.step()
        hist_lrs.append(fl)
        trail_lrs_x.append(xl[0]); trail_lrs_y.append(xl[1]); trail_lrs_z.append(fl)
        lrs_point.set_data([xl[0]], [xl[1]])
        lrs_point.set_3d_properties([fl])
        line_lrs.set_data(trail_lrs_x, trail_lrs_y)
