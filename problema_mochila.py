import math

import numpy as np

def calc_valor_resp(res, pesos, valores, capacidade):
    # Calcula o nível de satisfazibilidade de uma solução
    valor_total = np.sum(valores * res)
    peso_total = np.sum(pesos * res)
    if peso_total <= capacidade:
        penalidade = 0
    else:
        penalidade = 1000 * (peso_total - capacidade)
    return valor_total - penalidade


def sol_mochila(pesos, valores, capacidade, max_iter=1000, temp_ini=1000, alpha=0.95):
    n_itens = len(pesos)

    res_atual = np.random.randint(0, 2, size=n_itens)
    val_res = calc_valor_resp(res_atual, pesos, valores, capacidade)
    melhor_res = res_atual
    melhor_val = val_res

    temp = temp_ini
    for i in range(max_iter):
        prox_res = res_atual.copy()
        index = np.random.randint(0, n_itens)
        prox_res[index] = 1 - prox_res[index]

        prox_val = calc_valor_resp(prox_res, pesos, valores, capacidade)

        diff_val = prox_val - val_res
        if diff_val > 0 or np.random.rand() < math.exp(diff_val / temp):
            res_atual = prox_res
            val_res = prox_val

        if val_res > melhor_val:
            melhor_res = res_atual
            melhor_val = val_res

        temp *= alpha

        if i % (max_iter // 10) == 0:
            print(f'iteração {i}: melhor valor {melhor_val}')
    return melhor_res, melhor_val

def criar_valores(n_itens, min_peso, max_peso, min_val, max_val, capacidade=0.6):
    pesos = np.random.randint(min_peso, max_peso + 1, size=n_itens)
    valores = np.random.randint(min_val, max_val + 1, size=n_itens)

    peso_total = np.sum(pesos)
    capacidade = peso_total * capacidade
    return pesos, valores, capacidade

