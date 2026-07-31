# -*- coding: utf-8 -*-
"""
teste_conservacao_passo2.py
=============================

Mesmo espirito de `teste_conservacao.py` (Passo 1), agora incluindo a
categoria de gelo: a soma total de agua (vapor + nuvem + chuva + gelo)
na coluna, mais a precipitacao acumulada que ja saiu pela base, deve
permanecer CONSTANTE ao longo do tempo.

Rodar com:  python3 teste_conservacao_passo2.py
"""

import numpy as np
from microfisica.coluna_step2 import ColunaFaseGelo


def massa_total_agua(coluna):
    """Massa total de agua (vapor+nuvem+chuva+gelo) na coluna, kg/m^2."""
    return np.sum((coluna.qv + coluna.qc + coluna.qr + coluna.qi) * coluna.rho) * coluna.dz


def rodar_teste(tempo_total_s=1800.0, dt=2.0, tol_relativa=1.0e-6):
    coluna = ColunaFaseGelo(nz=80, dz=100.0, T_base=293.0, p_base=95000.0)
    k_base = int(1500 / coluna.dz)
    k_topo = int(6000 / coluna.dz)
    coluna.inserir_nuvem(k_base, k_topo, qc_valor=1.0e-3, Nc_valor=2.0e8)

    massa_inicial = massa_total_agua(coluna)
    coluna.integrar(tempo_total_s, dt=dt, salvar_a_cada=1.0e9)
    massa_final = massa_total_agua(coluna)
    precip_equiv = coluna.precip_superficie_mm  # kg/m^2

    balanco = massa_inicial - (massa_final + precip_equiv)
    erro_relativo = abs(balanco) / massa_inicial

    print(f"Massa total de agua inicial:         {massa_inicial:.8f} kg/m^2")
    print(f"Massa total de agua final (coluna):   {massa_final:.8f} kg/m^2")
    print(f"Precipitacao acumulada (equivalente): {precip_equiv:.8f} kg/m^2")
    print(f"Erro de balanco (inicial - final - precip): {balanco:.3e} kg/m^2")
    print(f"Erro relativo: {erro_relativo:.3e}")

    print(f"\nqc maximo final: {coluna.qc.max():.6e} kg/kg")
    print(f"qi maximo final: {coluna.qi.max():.6e} kg/kg")

    if erro_relativo < tol_relativa:
        print("\n[OK] O modelo CONSERVA massa de agua (com gelo) dentro da "
              f"tolerancia numerica esperada (< {tol_relativa:.0e}).")
    else:
        print("\n[FALHA] Violacao de conservacao de massa detectada! "
              "Revisar os processos de microfisica da fase gelo.")

    return erro_relativo


if __name__ == "__main__":
    rodar_teste()
