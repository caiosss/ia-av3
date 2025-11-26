# main_problemas_continuos.py
import numpy as np
import matplotlib.pyplot as plt
from algoritmos.algoritmos_busca import HillClimbing, LocalRandomSearch, GlobalRandomSearch

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

    hc.reset(epsilon=0.1, max_it=max_frames)
    lrs.reset(sigma=0.1, epsilon=0.1, max_it=max_frames)
    grs.reset(max_it=max_frames)

    hist_hc = []
    hist_lrs = []
    hist_grs = []

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
        hc_point.set_data([xh[0]], [xh[1]])
        hc_point.set_3d_properties([fh])
        hist_hc.append(fh)

        trail_hc_x.append(xh[0])
        trail_hc_y.append(xh[1])
        trail_hc_z.append(fh)
        line_hc.set_data(trail_hc_x, trail_hc_y)
        line_hc.set_3d_properties(trail_hc_z)

        xl, fl, _ = lrs.step()
        lrs_point.set_data([xl[0]], [xl[1]])
        lrs_point.set_3d_properties([fl])
        hist_lrs.append(fl)

        trail_lrs_x.append(xl[0])
        trail_lrs_y.append(xl[1])
        trail_lrs_z.append(fl)
        line_lrs.set_data(trail_lrs_x, trail_lrs_y)
        line_lrs.set_3d_properties(trail_lrs_z)

        xg, fg, _ = grs.step()
        grs_point.set_data([xg[0]], [xg[1]])
        grs_point.set_3d_properties([fg])
        hist_grs.append(fg)

        trail_grs_x.append(xg[0])
        trail_grs_y.append(xg[1])
        trail_grs_z.append(fg)
        line_grs.set_data(trail_grs_x, trail_grs_y)
        line_grs.set_3d_properties(trail_grs_z)

        plt.pause(0.001)

    plt.show()

    print("\n===== RESULTADOS =====")
    print("Hill Climbing:", hist_hc[-1])
    print("Local Random Search:", hist_lrs[-1])
    print("Global Random Search:", hist_grs[-1])

    plt.figure()
    plt.plot(hist_hc, 'r', label="HC")
    plt.plot(hist_lrs, 'g', label="LRS")
    plt.plot(hist_grs, 'b', label="GRS")
    plt.title(f"Históricos - {nome}")
    plt.xlabel("Iteração")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":

    for nome, funcao, dominio, modo in PROBLEMAS:
        print(f"\n### Executando {nome} ###")
        animar_algoritmos(funcao, dominio, modo, nome)
