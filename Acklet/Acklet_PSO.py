import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

# Função de Ackley
def ackley(x):
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + 20 + np.e

# Parâmetros do PSO
num_particulas = 30
dimensoes = 2
max_iter = 100
w = 0.7
c1 = 1.5
c2 = 1.5

# Preparação da visualização
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([ackley(np.array([xi, yi])) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

# Inicialização das partículas e velocidades
particulas = np.random.uniform(-5, 5, (num_particulas, dimensoes))
velocidades = np.zeros((num_particulas, dimensoes))
melhor_pos_particula = particulas.copy()
melhor_valor_particula = np.array([ackley(p) for p in particulas])
melhor_pos_global = melhor_pos_particula[np.argmin(melhor_valor_particula)]
melhor_valor_global = np.min(melhor_valor_particula)

# Arrays para histórico
historico_melhores = []
historico_piores = []

# Primeiro gráfico: Heatmap e partículas (animação)
def executar_animacao_pso():
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im = ax1.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis', alpha=0.7)
    scatter = ax1.scatter(particulas[:, 0], particulas[:, 1], c='red', s=50, label='Partículas')
    ax1.set_title("Evolução do PSO - Heatmap da Função de Ackley")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    plt.colorbar(im, ax=ax1, label='Valor da Ackley')
    ax1.legend()

    def update(frame):
        global particulas, velocidades, melhor_pos_particula, melhor_valor_particula, melhor_pos_global, melhor_valor_global
        
        # Atualiza velocidades e posições
        for i in range(num_particulas):
            r1, r2 = np.random.rand(), np.random.rand()
            velocidades[i] = (w * velocidades[i] +
                             c1 * r1 * (melhor_pos_particula[i] - particulas[i]) +
                             c2 * r2 * (melhor_pos_global - particulas[i]))
            particulas[i] += velocidades[i]
            
            # Atualiza melhores locais e global
            valor_atual = ackley(particulas[i])
            if valor_atual < melhor_valor_particula[i]:
                melhor_valor_particula[i] = valor_atual
                melhor_pos_particula[i] = particulas[i].copy()
                if valor_atual < melhor_valor_global:
                    melhor_valor_global = valor_atual
                    melhor_pos_global = particulas[i].copy()
        
        # Atualiza histórico
        fitness_atual = np.array([ackley(p) for p in particulas])
        historico_melhores.append(np.min(fitness_atual))
        historico_piores.append(np.max(fitness_atual))
        
        # Atualiza o scatter plot
        scatter.set_offsets(particulas)
        ax1.set_title(f"Iteração {frame + 1} - Melhor Fitness: {melhor_valor_global:.4f}")
        
        time.sleep(0.01)
        return scatter,

    ani = FuncAnimation(fig1, update, frames=max_iter, interval=300, blit=False, repeat=False)
    plt.show()
    return historico_melhores, historico_piores

# Segundo gráfico: Convergência (estático)
def plotar_convergencia(historico_melhores, historico_piores):
    plt.figure(figsize=(10, 6))
    plt.plot(historico_melhores, 'b-', label="Melhor Fitness", linewidth=2)
    plt.plot(historico_piores, 'g-', label="Pior Fitness", linewidth=2)
    plt.title("Convergência do PSO")
    plt.xlabel("Iteração")
    plt.ylabel("Valor da Ackley")
    plt.legend()
    plt.grid(True)
    plt.show()

# Execução sequencial
print("Executando animação do PSO...")
melhores, piores = executar_animacao_pso()

print("Mostrando gráfico de convergência...")
plotar_convergencia(melhores, piores)

print("Visualização concluída!")