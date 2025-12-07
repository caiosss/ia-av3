import numpy as np
import matplotlib.pyplot as plt

def distancia(p1, p2):
    return np.linalg.norm(p1 - p2)

def calcular_custo_rota(rota, pontos):
    total = 0
    for i in range(len(rota)):
        p1 = pontos[rota[i]]
        p2 = pontos[rota[(i+1) % len(rota)]]
        total += distancia(p1, p2)
    return total

class AlgoritmoGenetico:

    def __init__(self, pontos, pop_size=80, generations=500,
                 mutation_rate=0.01, tournament_size=5, elitismo=0):

        self.pontos = np.array(pontos)
        self.N = len(pontos)

        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitismo = elitismo

        self.population = []
        self.fitness = []

        self.best_solution = None
        self.best_cost = float("inf")

    def inicializar_populacao(self):
        self.population = []
        for _ in range(self.pop_size):
            perm = np.random.permutation(self.N)
            self.population.append(perm)

    def avaliar_populacao(self):
        self.fitness = []
        for ind in self.population:
            custo = calcular_custo_rota(ind, self.pontos)
            self.fitness.append(custo)

            if custo < self.best_cost:
                self.best_cost = custo
                self.best_solution = ind.copy()

    def torneio(self):
        indices = np.random.choice(len(self.population), self.tournament_size)
        melhor = None
        melhor_custo = float("inf")
        for i in indices:
            custo = self.fitness[i]
            if custo < melhor_custo:
                melhor = self.population[i].copy()
                melhor_custo = custo
        return melhor

    def crossover(self, p1, p2):
        n = len(p1)
        a, b = sorted(np.random.choice(n, 2, replace=False))

        filho = [-1] * n
        filho[a:b] = p1[a:b]

        pos = 0
        for gene in p2:
            if gene not in filho:
                while filho[pos] != -1:
                    pos += 1
                filho[pos] = gene

        return np.array(filho)

    def mutacao(self, ind):
        if np.random.random() < self.mutation_rate:
            i, j = np.random.choice(len(ind), 2, replace=False)
            ind[i], ind[j] = ind[j], ind[i]

    def executar(self, animar=False):

        self.inicializar_populacao()
        self.avaliar_populacao()

        historico_best = []

        if animar:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.ion()

            ax.set_facecolor("#f8f8f8")
            fig.set_facecolor("white")

            ax.view_init(elev=28, azim=-60)

        for gen in range(self.generations):

            nova_pop = []

            if self.elitismo > 0:
                melhores_idx = np.argsort(self.fitness)[:self.elitismo]
                for idx in melhores_idx:
                    nova_pop.append(self.population[idx].copy())

            while len(nova_pop) < self.pop_size:
                p1 = self.torneio()
                p2 = self.torneio()
                filho = self.crossover(p1, p2)
                self.mutacao(filho)
                nova_pop.append(filho)

            self.population = nova_pop
            self.avaliar_populacao()

            historico_best.append(self.best_cost)

            if animar:
                ax.cla()
                ax.set_title(f"Geração {gen} — Melhor custo: {self.best_cost:.2f}",
                             fontweight='bold')

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                ax.set_xlim(np.min(self.pontos[:,0]), np.max(self.pontos[:,0]))
                ax.set_ylim(np.min(self.pontos[:,1]), np.max(self.pontos[:,1]))
                ax.set_zlim(np.min(self.pontos[:,2]), np.max(self.pontos[:,2]))

                ax.scatter(self.pontos[:,0],
                           self.pontos[:,1],
                           self.pontos[:,2],
                           c=np.linspace(0,1,self.N),
                           cmap="viridis",
                           s=40,
                           alpha=0.9,
                           edgecolor="black")

                caminho = self.best_solution
                xs = self.pontos[caminho, 0]
                ys = self.pontos[caminho, 1]
                zs = self.pontos[caminho, 2]

                ax.plot(xs, ys, zs,
                        color="crimson",
                        linewidth=2,
                        label="Melhor Rota")

                ax.plot([xs[-1], xs[0]],
                        [ys[-1], ys[0]],
                        [zs[-1], zs[0]],
                        color="black",
                        linewidth=2)

                ax.scatter(xs, ys, zs,
                           c="crimson",
                           s=25)

                plt.pause(0.01)

        if animar:
            plt.ioff()
            plt.show()


        return historico_best, self.best_solution, self.best_cost
