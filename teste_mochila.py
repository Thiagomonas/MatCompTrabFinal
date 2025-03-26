from problema_mochila import *

pesos, valores, capacidade = criar_valores(100, 5, 50, 1, 15)
historico = sol_mochila_historico_valores(pesos, valores, capacidade, n_reps=5)

gerar_grafico(historico)
