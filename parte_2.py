import time
from algoritmos.algoritmos_busca  import TemperaSimulada
import matplotlib.pyplot as plt

def canonical(x):
    sols = []
    x = list(x)
    sols.append(tuple(x))
    sols.append(tuple(x[::-1]))
    rot90 = [0]*8
    for col in range(8):
        rot90[x[col]] = 7-col
    sols.append(tuple(rot90))
    sols.append(tuple(rot90[::-1]))
    return min(sols)
if __name__ == "__main__":
    print("=== Buscando TODAS as 92 soluções do problema das 8 rainhas ===")
    inicio = time.time()
    solucoes = set()
    tentativas = 0
    while len(solucoes) < 92:
        tentativas += 1
        ts = TemperaSimulada(max_it=5000, T=1.0, alpha=0.98)
        sol, valor = ts.search()
        if valor == 28:
            sol_can = canonical(sol)
            if sol_can not in solucoes:
                print(f"Nova solução encontrada! ({len(solucoes)+1}/92)")
                print(" →", sol)
                solucoes.add(sol_can)
    fim = time.time()
    plt.figure(figsize=(10, 5))
    plt.plot(ts.historico, linewidth=2)
    plt.title("Histórico da Função Objetivo - Tempera Simulada (8 Rainhas)")
    plt.xlabel("Iteração")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.ylim(0, 28)
    plt.show()
    print("\n=== RESULTADO FINAL ===")
    print("Total de soluções distintas encontradas:", len(solucoes))
    print("Tentativas executadas:", tentativas)
    print(f"Tempo total: {fim - inicio:.2f} segundos")
