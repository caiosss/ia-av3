import numpy as np
from algoritmos.algoritmo_genetico import AlgoritmoGenetico
import matplotlib.pyplot as plt

pontos = np.loadtxt("CaixeiroGruposGA.csv", delimiter=",", skiprows=1)

ga = AlgoritmoGenetico(
    pontos,
    pop_size=80,
    generations=400,
    mutation_rate=0.01,
    tournament_size=4,
    elitismo=1
)

historico, melhor_rota, melhor_custo = ga.executar(animar=True)

print("Melhor custo encontrado:", melhor_custo)
print("Melhor rota encontrada:", melhor_rota)

plt.plot(historico)
plt.title("Histórico do Melhor Custo")
plt.xlabel("Geração")
plt.ylabel("Custo")
plt.grid()
plt.show()
