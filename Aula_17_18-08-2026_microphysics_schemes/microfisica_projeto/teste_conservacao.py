# -*- coding: utf-8 -*-
"""
teste_conservacao.py
======================

Teste de verificacao (sanity check) do modelo de microfisica: a soma
total de agua (vapor + nuvem + chuva) na coluna, MAIS a precipitacao
acumulada que ja saiu pela base, deve permanecer CONSTANTE ao longo do
tempo (conservacao de massa de agua) -- nenhum processo de microfisica
cria ou destroi massa, apenas converte entre as diferentes categorias
(vapor <-> nuvem <-> chuva) ou remove definitivamente da coluna via
precipitacao de superficie.

Este e o tipo de teste que qualquer esquema de microfisica "bulk" deve
satisfazer (e um dos testes numericos que Morrison & Gettelman 2008
reportam na Parte I do artigo, secao 3 "Numerical tests").

Rodar com:  python3 teste_conservacao.py
"""

import numpy as np
from microfisica.coluna_step1 import ColunaChuvaQuente
from microfisica.constantes import rho_w


def massa_total_agua(coluna):
    """Massa total de agua (vapor+nuvem+chuva) na coluna, kg/m^2."""
    return np.sum((coluna.qv + coluna.qc + coluna.qr) * coluna.rho) * coluna.dz


def massa_precip_equivalente(coluna):
    """Converte a precipitacao acumulada (mm de lamina d'agua) para
    kg/m^2 (1 mm de lamina = 1 kg/m^2, pela definicao de mm de chuva)."""
    return coluna.precip_superficie_mm  # numericamente ja equivalente a kg/m^2


def rodar_teste(tempo_total_s=3600.0, dt=5.0, tol_relativa=1.0e-6):
    coluna = ColunaChuvaQuente(nz=40, dz=100.0, T_base=293.0, p_base=95000.0)
    k_base = int(1000 / coluna.dz)
    k_topo = int(2500 / coluna.dz)
    coluna.inserir_nuvem(k_base, k_topo, qc_valor=1.5e-3, Nc_valor=1.0e8)

    massa_inicial = massa_total_agua(coluna)
    coluna.integrar(tempo_total_s, dt=dt, salvar_a_cada=1.0e9)
    massa_final = massa_total_agua(coluna)
    precip_equiv = massa_precip_equivalente(coluna)

    # balanco: massa_inicial deve ser igual a massa_final (na coluna) +
    # massa que ja saiu como precipitacao de superficie
    balanco = massa_inicial - (massa_final + precip_equiv)
    erro_relativo = abs(balanco) / massa_inicial

    print(f"Massa total de agua inicial:         {massa_inicial:.8f} kg/m^2")
    print(f"Massa total de agua final (coluna):   {massa_final:.8f} kg/m^2")
    print(f"Precipitacao acumulada (equivalente): {precip_equiv:.8f} kg/m^2")
    print(f"Erro de balanco (inicial - final - precip): {balanco:.3e} kg/m^2")
    print(f"Erro relativo: {erro_relativo:.3e}")

    if erro_relativo < tol_relativa:
        print("\n[OK] O modelo CONSERVA massa de agua dentro da tolerancia "
              f"numerica esperada (< {tol_relativa:.0e}).")
    else:
        print("\n[FALHA] Violacao de conservacao de massa detectada! "
              "Revisar os processos de microfisica.")

    return erro_relativo


if __name__ == "__main__":
    rodar_teste()
