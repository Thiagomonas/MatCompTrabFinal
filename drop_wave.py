import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# Definição da função Drop-Wave
def drop_wave(x, y):
    numerador = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
    denominador = 0.5 * (x**2 + y**2) + 2
    return -numerador / denominador

# Parâmetros gerais
tam_populacao = num_particulas = 50
iteracoes = 100
x_min, x_max = -5.12, 5.12
v_maz = 1.0
W, C1, C2 = 0.7, 1.5, 1.5  # PSO
F, CR = 0.8, 0.9  # DE

# PSO - Inicialização das partículas
class Particula:
    def __init__(self):
        self.x = np.random.uniform(x_min, x_max)
        self.y = np.random.uniform(x_min, x_max)        
        self.vx = np.random.uniform(-v_maz, v_maz)
        self.vy = np.random.uniform(-v_maz, v_maz)
        self.melhor_x = self.x
        self.melhor_y = self.y
        self.melhor_fitness = drop_wave(self.x, self.y)

swarm = [Particula() for _ in range(num_particulas)]
# melhor_particula = min(swarm, key=lambda p: p.melhor_fitness)
# g_melhor_x, g_melhor_y = melhor_particula.melhor_x, melhor_particula.melhor_y
# g_melhor_fitness = melhor_particula.melhor_fitness

g_melhor_x, g_melhor_y = min(swarm, key=lambda p: p.melhor_fitness).melhor_x, min(swarm, key=lambda p: p.melhor_fitness).melhor_y
g_melhor_fitness = drop_wave(g_melhor_x, g_melhor_y)

# DE - Inicialização da população
populacao = np.random.uniform(x_min, x_max, (tam_populacao, 2))
fitness = np.apply_along_axis(lambda p: drop_wave(p[0], p[1]), 1, populacao)

# Execução do PSO e DE
for _ in range(iteracoes):
    # PSO
    for p in swarm:
        p.vx = W * p.vx + C1 * np.random.rand() * (p.melhor_x - p.x) + C2 * np.random.rand() * (g_melhor_x - p.x)
        p.vy = W * p.vy + C1 * np.random.rand() * (p.melhor_y - p.y) + C2 * np.random.rand() * (g_melhor_y - p.y)
        p.x = np.clip(p.x + p.vx, x_min, x_max)
        p.y = np.clip(p.y + p.vy, x_min, x_max)
        fitness_p = drop_wave(p.x, p.y)
        if fitness_p < p.melhor_fitness:
            p.melhor_fitness = fitness_p
            p.melhor_x = p.x
            p.melhor_y = p.y
        if fitness_p < g_melhor_fitness:
            g_melhor_fitness = fitness_p
            g_melhor_x = p.x
            g_melhor_y = p.y
    
    # DE
    for i in range(tam_populacao):
        idxs = [idx for idx in range(tam_populacao) if idx != i]
        a, b, c = populacao[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + F * (b - c), x_min, x_max)
        cross_points = np.random.rand(2) < CR
        trial = np.where(cross_points, mutant, populacao[i])
        trial_fitness = drop_wave(trial[0], trial[1])
        if trial_fitness < fitness[i]:
            populacao[i] = trial
            fitness[i] = trial_fitness

melhor_idx_de = np.argmin(fitness)
melhor_x_de, melhor_y_de = populacao[melhor_idx_de]

# Gráficos lado a lado
x_vals = np.linspace(x_min, x_max, 100)
y_vals = np.linspace(x_min, x_max, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = drop_wave(X, Y)

fig = plt.figure(figsize=(14, 6))

# PSO Plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)
ax1.scatter([p.x for p in swarm], [p.y for p in swarm], [drop_wave(p.x, p.y) for p in swarm], color='yellow', label='Partículas')
ax1.scatter(g_melhor_x, g_melhor_y, drop_wave(g_melhor_x, g_melhor_y), color='red', marker='*', s=100, label='Melhor Global')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Fitness')
ax1.set_title(f'PSO Otimização - Drop-Wave Função\nMelhor Fitness: {g_melhor_fitness:.5f}')
ax1.legend()

# DE Plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)
ax2.scatter(populacao[:, 0], populacao[:, 1], fitness, color='yellow', label='População')
ax2.scatter(melhor_x_de, melhor_y_de, drop_wave(melhor_x_de, melhor_y_de), color='red', marker='*', s=100, label='Melhor Solução')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Fitness')
ax2.set_title(f'DE Otimização - Drop-Wave Função\nMelhor Fitness: {np.min(fitness):.5f}')
ax2.legend()

plt.show()
