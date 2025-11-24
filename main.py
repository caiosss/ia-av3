import numpy as np
import matplotlib.pyplot as plt
from algoritmos.algoritmos_busca import LocalRandomSearch
from algoritmos.algoritmos_busca import HillClimbing
from algoritmos.algoritmos_busca import GlobalRandomSearch

# hc = HillClimbing()
# melhor, todos = hc.search(R=30, epsilon=0.1, max_it=500, t_sem_melhoria=50)

# points = np.random.uniform(low=0, high=100, size=(20,2))

lrs = LocalRandomSearch(lim_inf=-2, lim_sup=4)
lrs.search()

# grs = GlobalRandomSearch(lim_inf=-2, lim_sup=4)
# grs.search()