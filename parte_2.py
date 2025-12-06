from algoritmos.algoritmos_busca  import TemperaSimulada
import matplotlib.pyplot as plt


def rotate_90(sol):
    new = [0]*8
    for col in range(8):
        row = sol[col]
        new_col = row
        new_row = 7 - col
        new[new_col] = new_row
    return tuple(new)

def rotate_180(sol):
    return rotate_90(rotate_90(sol))

def rotate_270(sol):
    return rotate_90(rotate_180(sol))

def reflect(sol):
    return tuple(7 - row for row in sol)
def canonical(sol):
    return tuple(sol)   


if __name__ == "__main__":
    print("=== Buscando TODAS as 92 soluções do problema das 8 rainhas (DEBUG) ===")
    solucoes = set()
    mapa = {}
    tentativas = 0
    last_print = 0
    PRINT_EVERY = 200
    LIMITE_TENTATIVAS = 200000

    while len(solucoes) < 92 and tentativas < LIMITE_TENTATIVAS:
        tentativas += 1
        ts = TemperaSimulada(max_it=4000, T=12.0, alpha=0.9995)
        sol, valor = ts.search()
        if sol is None:
            continue
        if valor == 28:
            sol_can = canonical(sol)     
            transforms = []             
            if sol_can not in solucoes:
                solucoes.add(sol_can)
                mapa.setdefault(sol_can, set()).add(tuple(sol))
                idx = len(solucoes)
                print(f"Nova solução encontrada! ({idx}/92)")
                print(" →", sol)
        if tentativas % 1000 == 0:
            print(f"tentativas={tentativas}  soluções={len(solucoes)}  tempo={(time.time()-inicio):.1f}s")

    print("\n=== RESULTADO FINAL ===")
    print("Total de soluções distintas encontradas:", len(solucoes))
    print("Tentativas executadas:", tentativas)
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(ts.historico)
        plt.title("Histórico da última execução (f(x))")
        plt.xlabel("Iteração")
        plt.ylabel("f(x)")
        plt.ylim(0, 28)
        plt.grid(True)
        plt.show()
    except Exception:
        pass