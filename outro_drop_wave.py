import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

# Função Drop Wave - função de teste para otimização
def onda_drop(ponto=[1, 2]):
    '''
    Calcula Drop Wave para um ponto 2D.
    Recebe:
    ponto -> array com coordenadas [x1, x2]
    Retorna:
    Valor da função (negativo para transformar em problema de minimização)
    '''
    x1, x2 = ponto
    numerador = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    denominador = 0.5 * (x1**2 + x2**2) + 2
    return -numerador / denominador

limites = [(-5.12, 5.12), (-5.12, 5.12)]  # Limites do espaço de busca para x1 e x2
max_iteracoes = 200  # Número máximo de iterações

# Implementação do PSO
class PSO:
    def __init__(self, funcao, limites, num_particulas=50, max_iter=100, inercia=0.7, c1=1.5, c2=1.5):
        '''
        Inicializa o otimizador PSO.
        
        funcao -> função objetivo a ser minimizada
        limites -> limites do espaço de busca
        num_particulas -> número de partículas no enxame
        max_iter -> máximo de iterações
        inercia -> peso de inércia
        c1 -> coeficiente cognitivo
        c2 -> coeficiente social
        '''
        self.funcao = funcao
        self.limites = limites
        self.num_particulas = num_particulas
        self.max_iter = max_iter
        self.inercia = inercia
        self.c1 = c1
        self.c2 = c2
        
        # Dimensões do problema (2D para Drop Wave)
        self.dim = len(limites)
        
        # Inicializa partículas com posições aleatórias dentro dos limites
        self.particulas = np.random.uniform(
            low=[limite[0] for limite in limites],
            high=[limite[1] for limite in limites],
            size=(num_particulas, self.dim)
        )
        
        # Inicializa velocidades como 0
        self.velocidades = np.zeros((num_particulas, self.dim))
        
        # Melhores posições individuais
        self.melhores_pos = self.particulas.copy()
        self.melhores_valores = np.array([funcao(p) for p in self.particulas])
        
        # Melhor posição global
        self.melhor_global_idx = np.argmin(self.melhores_valores)
        self.melhor_global_pos = self.melhores_pos[self.melhor_global_idx].copy()
        self.melhor_global_valor = self.melhores_valores[self.melhor_global_idx]
        
        # Históricos de posições e valores
        self.historico = [self.particulas.copy()]  # Armazena todas as posições
        self.historico_melhor = [self.melhor_global_valor]  # Armazena melhores valores
        self.historico_pior = [np.max([funcao(p) for p in self.particulas])]  # Armazena piores valores
    
    def otimizar(self):
        '''
        Algoritmo PSO.
        Retorna:
        melhor_pos -> melhor posição encontrada
        melhor_valor -> melhor valor da função encontrado
        '''
        for _ in range(self.max_iter):
            # Gera números aleatórios para os componentes
            rand1 = np.random.rand(self.num_particulas, self.dim)
            rand2 = np.random.rand(self.num_particulas, self.dim)
            
            # Atualiza velocidades (equação do PSO)
            self.velocidades = (self.inercia * self.velocidades +
                              self.c1 * rand1 * (self.melhores_pos - self.particulas) +
                              self.c2 * rand2 * (self.melhor_global_pos - self.particulas))
            
            # Atualiza posições
            self.particulas += self.velocidades
            
            # Mantém partículas dentro dos limites
            for i in range(self.dim):
                self.particulas[:, i] = np.clip(
                    self.particulas[:, i], 
                    self.limites[i][0], 
                    self.limites[i][1]
                )
            
            # Avalia a função nas novas posições
            valores_atuais = np.array([self.funcao(p) for p in self.particulas])
            
            # Atualiza melhores posições individuais
            melhorou = valores_atuais < self.melhores_valores
            self.melhores_pos[melhorou] = self.particulas[melhorou]
            self.melhores_valores[melhorou] = valores_atuais[melhorou]
            
            # Atualiza melhor global
            idx_melhor_atual = np.argmin(self.melhores_valores)
            if self.melhores_valores[idx_melhor_atual] < self.melhor_global_valor:
                self.melhor_global_pos = self.melhores_pos[idx_melhor_atual].copy()
                self.melhor_global_valor = self.melhores_valores[idx_melhor_atual]
            
            # Armazena histórico
            self.historico.append(self.particulas.copy())
            self.historico_melhor.append(self.melhor_global_valor)
            self.historico_pior.append(np.max(valores_atuais))
        
        return self.melhor_global_pos, self.melhor_global_valor

# Implementação do DE
class DE:
    def __init__(self, funcao, limites, tam_pop=50, max_iter=100, F=0.8, CR=0.9):
        '''
        Inicializa o otimizador DE.
        Recebe:
        funcao -> função objetivo a ser minimizada
        limites -> limites do espaço de busca
        tam_pop -> tamanho da população
        max_iter -> máximo de iterações
        F -> fator de mutação
        CR -> taxa de crossover
        '''
        self.funcao = funcao
        self.limites = limites
        self.tam_pop = tam_pop
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        
        # Dimensões do problema
        self.dim = len(limites)
        
        # Inicializa população aleatória
        self.populacao = np.random.uniform(
            low=[limite[0] for limite in limites],
            high=[limite[1] for limite in limites],
            size=(tam_pop, self.dim)
        )
        
        # Avaliação inicial
        self.aptidao = np.array([funcao(ind) for ind in self.populacao])
        
        # Históricos
        self.historico_melhor = [np.min(self.aptidao)]
        self.historico_pior = [np.max(self.aptidao)]
    
    def otimizar(self):
        '''
        Executa DE.
        Retorna:
        melhor_individuo -> melhor solução encontrada
        melhor_aptidao -> melhor valor da função encontrado
        '''
        for _ in range(self.max_iter):
            nova_pop = []
            nova_aptidao = []
            
            for i in range(self.tam_pop):
                # Seleciona 3 indivíduos diferentes
                indices = [idx for idx in range(self.tam_pop) if idx != i]
                a, b, c = self.populacao[np.random.choice(indices, 3, replace=False)]
                
                # Operação de mutação
                mutante = a + self.F * (b - c)
                mutante = np.clip(mutante, [b[0] for b in self.limites], [b[1] for b in self.limites])
                
                # Operação de crossover
                pontos_cross = np.random.rand(self.dim) < self.CR
                if not np.any(pontos_cross):
                    pontos_cross[np.random.randint(0, self.dim)] = True
                
                # Cria indivíduo de teste
                teste = np.where(pontos_cross, mutante, self.populacao[i])
                
                # Seleção
                aptidao_teste = self.funcao(teste)
                if aptidao_teste < self.aptidao[i]:
                    nova_pop.append(teste)
                    nova_aptidao.append(aptidao_teste)
                else:
                    nova_pop.append(self.populacao[i])
                    nova_aptidao.append(self.aptidao[i])
            
            # Atualiza população
            self.populacao = np.array(nova_pop)
            self.aptidao = np.array(nova_aptidao)
            
            # Armazena histórico
            self.historico_melhor.append(np.min(self.aptidao))
            self.historico_pior.append(np.max(self.aptidao))
        
        # Retorna o melhor indivíduo encontrado
        melhor_idx = np.argmin(self.aptidao)
        return self.populacao[melhor_idx], self.aptidao[melhor_idx]


# Instancia e executa PSO
otimizador_pso = PSO(onda_drop, limites, num_particulas=50, max_iter=max_iteracoes)
melhor_pos_pso, melhor_valor_pso = otimizador_pso.otimizar()

# Instancia e executa DE
otimizador_de = DE(onda_drop, limites, tam_pop=50, max_iter=max_iteracoes)
melhor_pos_de, melhor_valor_de = otimizador_de.otimizar()

# Exibe resultados
print(f"PSO - Melhor solução encontrada: {melhor_pos_pso} -> Valor da: {melhor_valor_pso}")
print(f"DE - Melhor solução encontrada: {melhor_pos_de} -> Valor da: {melhor_valor_de}")

# Configura figura para os gráficos de convergência
plt.figure(figsize=(12, 5))

# Gráfico de convergência do PSO
plt.subplot(1, 2, 1)
plt.plot(otimizador_pso.historico_melhor, label='Melhor solução')
plt.plot(otimizador_pso.historico_pior, 'r--', alpha=0.3, label='Pior solução')
plt.title('Convergência do PSO')
plt.xlabel('Iteração')
plt.ylabel('Valor da função')
plt.legend()

# Gráfico de convergência do DE
plt.subplot(1, 2, 2)
plt.plot(otimizador_de.historico_melhor, label='Melhor solução')
plt.plot(otimizador_de.historico_pior, 'r--', alpha=0.3, label='Pior solução')
plt.title('Convergência do DE')
plt.xlabel('Iteração')
plt.ylabel('Valor da função')
plt.legend()

plt.tight_layout()
plt.show()

## Heatmap do movimento das partículas no PSO

# Prepara grid para o heatmap
x = np.linspace(limites[0][0], limites[0][1], 100)
y = np.linspace(limites[1][0], limites[1][1], 100)
X, Y = np.meshgrid(x, y)

# Calcula valores da função para o heatmap
Z = np.array([onda_drop([xi, yi]) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

# Seleciona iterações para visualização
iteracoes_vis = np.linspace(0, len(otimizador_pso.historico)-1, 10, dtype=int)

# Cria figura com os heatmaps
plt.figure(figsize=(15, 10))
for i, iter_num in enumerate(iteracoes_vis):
    plt.subplot(3, 4, i+1)
    plt.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
    plt.colorbar()
    plt.scatter(
        otimizador_pso.historico[iter_num][:, 0], 
        otimizador_pso.historico[iter_num][:, 1], 
        c='r', s=10, alpha=0.7
    )
    plt.title(f'Iteração {iter_num}')
    plt.xlim(limites[0])
    plt.ylim(limites[1])

plt.suptitle('Movimento das Partículas no PSO ao Longo das Iterações')
plt.tight_layout()
plt.show()