# -*- coding: utf-8 -*-
"""
teste_conservacao_passo3.py
=============================

Mesmo espirito de `teste_conservacao.py` e `teste_conservacao_passo2.py`,
agora incluindo neve e graupel: a soma total de agua (vapor + nuvem +
chuva + gelo + neve + graupel) na coluna, mais a precipitacao
acumulada que ja saiu pela base, deve permanecer CONSTANTE.

Este teste tambem inclui uma pequena quantidade de chuva pre-existente
numa regiao fria da coluna, para exercitar o congelamento heterogeneo
de chuva (Pgfzr -> graupel) alem dos processos ja testados no Passo 2.

Rodar com:  python3 teste_conservacao_passo3.py
"""

import numpy as np
from microfisica.coluna_step3 import ColunaFaseMista


def massa_total_agua(coluna):
    """Massa total de agua (todas as categorias) na coluna, kg/m^2."""
    return np.sum(
        (coluna.qv + coluna.qc + coluna.qr + coluna.qi + coluna.qs + coluna.qg)
        * coluna.rho
    ) * coluna.dz


def rodar_teste(tempo_total_s=1800.0, dt=2.0, tol_relativa=1.0e-6):
    coluna = ColunaFaseMista(nz=80, dz=100.0, T_base=293.0, p_base=95000.0)
    k_base = int(1500 / coluna.dz)
    k_topo = int(6000 / coluna.dz)
    coluna.inserir_nuvem(k_base, k_topo, qc_valor=1.0e-3, Nc_valor=2.0e8)

    # chuva pre-existente numa regiao fria (~-9 graus C), para testar Pgfzr
    k_fria = int(4500 / coluna.dz)
    coluna.qr[k_fria] = 5.0e-4
    coluna.Nr[k_fria] = 5.0e5

    massa_inicial = massa_total_agua(coluna)
    coluna.integrar(tempo_total_s, dt=dt, salvar_a_cada=1.0e9)
    massa_final = massa_total_agua(coluna)
    precip_equiv = coluna.precip_superficie_mm

    balanco = massa_inicial - (massa_final + precip_equiv)
    erro_relativo = abs(balanco) / massa_inicial

    print(f"Massa total de agua inicial:         {massa_inicial:.8f} kg/m^2")
    print(f"Massa total de agua final (coluna):   {massa_final:.8f} kg/m^2")
    print(f"Precipitacao acumulada (equivalente): {precip_equiv:.8f} kg/m^2")
    print(f"Erro de balanco (inicial - final - precip): {balanco:.3e} kg/m^2")
    print(f"Erro relativo: {erro_relativo:.3e}")

    print(f"\nqc maximo final: {coluna.qc.max():.6e} kg/kg")
    print(f"qi maximo final: {coluna.qi.max():.6e} kg/kg")
    print(f"qs maximo final: {coluna.qs.max():.6e} kg/kg")
    print(f"qg maximo final: {coluna.qg.max():.6e} kg/kg")

    # checagem extra: nenhum valor negativo ou NaN em nenhuma variavel
    problemas = []
    for nome in ["qv", "qc", "Nc", "qr", "Nr", "qi", "Ni", "qs", "Ns", "qg", "Ng"]:
        arr = getattr(coluna, nome)
        if np.isnan(arr).any() or (arr < 0).any():
            problemas.append(nome)

    if erro_relativo < tol_relativa and not problemas:
        print("\n[OK] O modelo CONSERVA massa de agua (com neve e graupel) "
              f"dentro da tolerancia numerica esperada (< {tol_relativa:.0e}), "
              "sem valores negativos ou NaN em nenhuma variavel.")
    else:
        print("\n[FALHA] Problema detectado!")
        if problemas:
            print("Variaveis com NaN/negativo:", problemas)

    return erro_relativo


if __name__ == "__main__":
    rodar_teste()
