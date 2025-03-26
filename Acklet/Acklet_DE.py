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

# Parâmetros da Evolução Diferencial
num_individuos = 30
dimensoes = 2
max_iter = 100
F = 0.8
CR = 0.9

# Preparação da visualização
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([ackley(np.array([xi, yi])) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

# Primeiro gráfico: Heatmap e indivíduos (animação)
def executar_animacao_de():
    # Inicialização da população (agora dentro da função)
    populacao = np.random.uniform(-5, 5, (num_individuos, dimensoes))
    fitness = np.array([ackley(individuo) for individuo in populacao])
    melhor_individuo = populacao[np.argmin(fitness)]
    melhor_fitness = np.min(fitness)
    
    # Arrays para histórico
    historico_melhores = [melhor_fitness]
    historico_piores = [np.max(fitness)]

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im = ax1.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis', alpha=0.7)
    scatter = ax1.scatter(populacao[:, 0], populacao[:, 1], c='red', s=50, label='Indivíduos')
    ax1.set_title("Evolução Diferencial - Heatmap da Função de Ackley")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    plt.colorbar(im, ax=ax1, label='Valor da Ackley')
    ax1.legend()

    def update(frame):
        nonlocal populacao, fitness, melhor_individuo, melhor_fitness, historico_melhores, historico_piores
        
        nova_populacao = np.zeros_like(populacao)
        novos_fitness = np.zeros(num_individuos)
        
        for i in range(num_individuos):
            # Seleciona 3 indivíduos diferentes
            idxs = [idx for idx in range(num_individuos) if idx != i]
            a, b, c = populacao[np.random.choice(idxs, 3, replace=False)]
            
            # Operação de mutação
            mutante = a + F * (b - c)
            
            # Operação de crossover
            crossover_points = np.random.rand(dimensoes) < CR
            if not np.any(crossover_points):
                crossover_points[np.random.randint(0, dimensoes)] = True
                
            trial = np.where(crossover_points, mutante, populacao[i])
            
            # Seleção
            trial_fitness = ackley(trial)
            if trial_fitness < fitness[i]:
                nova_populacao[i] = trial
                novos_fitness[i] = trial_fitness
            else:
                nova_populacao[i] = populacao[i]
                novos_fitness[i] = fitness[i]
        
        populacao = nova_populacao
        fitness = novos_fitness
        
        # Atualiza o melhor indivíduo
        melhor_fitness_atual = np.min(fitness)
        if melhor_fitness_atual < melhor_fitness:
            melhor_fitness = melhor_fitness_atual
            melhor_individuo = populacao[np.argmin(fitness)]
        
        # Atualiza histórico
        historico_melhores.append(melhor_fitness)
        historico_piores.append(np.max(fitness))
        
        # Atualiza o scatter plot
        scatter.set_offsets(populacao)
        ax1.set_title(f"Iteração {frame + 1} - Melhor Fitness: {melhor_fitness:.4f}")
        
        time.sleep(0.05)  # Reduzido para 50ms
        return scatter,

    ani = FuncAnimation(fig1, update, frames=max_iter, interval=300, blit=False, repeat=False)
    plt.show()
    return historico_melhores, historico_piores

# Segundo gráfico: Convergência 
def plotar_convergencia(historico_melhores, historico_piores):
    plt.figure(figsize=(10, 6))
    plt.plot(historico_melhores, 'b-', label="Melhor Fitness", linewidth=2)
    plt.plot(historico_piores, 'g-', label="Pior Fitness", linewidth=2)
    plt.title("Convergência da Evolução Diferencial")
    plt.xlabel("Iteração")
    plt.ylabel("Valor da Ackley")
    plt.legend()
    plt.grid(True)
    plt.show()

# Execução sequencial
print("Executando animação da Evolução Diferencial...")
melhores, piores = executar_animacao_de()

print("Mostrando gráfico de convergência...")
plotar_convergencia(melhores, piores)

print("Visualização concluída!")