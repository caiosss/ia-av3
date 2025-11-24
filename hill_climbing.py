import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-(x**2))

def f(X,Y):
    return np.exp(-(X**2+Y**2)) + 2*np.exp(-((X-2)**2+(Y-2)**2))

lim_inf = -2
lim_sup = 4

x_axis = np.linspace(lim_inf,lim_sup,500)

X,Y = np.meshgrid(x_axis,x_axis)

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y, f(X,Y),rstride=20,cstride=20,alpha=.3,edgecolor='k')


#Busca pela subida de encosta (Hill Climbing)
epsilon = .1 #Tamanho da vizinhança (epsilon)
max_it = 1000
max_vizinhos = 20
x_opt = np.random.uniform(lim_inf,lim_sup,size=(2,))
f_opt = f(*x_opt)
historico  = [f_opt]
ax.scatter(*x_opt, f_opt, c='r',s=40)

it = 0
melhoria = True
while it < max_it and melhoria:
    melhoria = False
    for j in range(max_vizinhos):
        #Perturbação do ótimo!
        x_cand = np.random.uniform(low=x_opt-epsilon, high= x_opt + epsilon)
        for i,x in enumerate(x_cand):
            if x < lim_inf:
                x_cand[i] = lim_inf
            if x > lim_sup:
                x_cand[i] = lim_sup
        
        f_cand = f(*x_cand)
        historico.append(f_opt)
        if f_cand > f_opt:
            x_opt = x_cand
            f_opt = f_cand
            melhoria = True
            plt.pause(.01)
            ax.scatter(*x_opt, f_opt, c='r',s=40)
            break
    it+=1
    


ax.scatter(*x_opt, f_opt, c='g',marker="*",s=250)
plt.figure(2)
plt.plot(historico)
plt.grid()
plt.show()

bp=1