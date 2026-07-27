# -*- coding: utf-8 -*-
"""
MODELO DE SUPERFICIE  SOLO NU  CICLO DIURNO
====================================================================
Integra sequencialmente os modulos 0-7, seguindo a MESMA ORDEM dos
slides "MET-576-4_Parte_1_Conceitos_Surface_Model":

    0. Forcantes atmosfericas       (SWd, LWd, Tr, er, Ur)
    1. Saldo de radiacao            Rn = (1-a)SWd + LWd - eps*sigma*T^4
    2. Fluxo de calor no solo       G  = (Cs*D/tau_d)*(T-Td)
    3. Resistencia aerodinamica     ra
    4. Fluxo de momentum            tau (diagnostico, nao afeta T)
    5. Fluxo de calor sensivel      H  = rho*cp*(T-Tr)/ra
    6. Fluxo de calor latente       LE = (rho*cp/gamma)*(h*es(T)-er)/(ra+rsoil)
    7. Balanco de agua do solo      d(theta)/dt = -E/D

Equacao prognostica de fechamento do sistema (slide "Hipoteses*"):

    Cs*D * dT/dt = Rn - G - H - LE

Integracao temporal: Euler explicito, passo dt pequeno (300 s),
por varios dias (spin-up) ate o ciclo diurno se estabilizar;
plota-se o ultimo dia.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from modulo0_forcantes import (
    radiacao_onda_curta_incidente,
    temperatura_referencia,
    radiacao_onda_longa_incidente,
    umidade_relativa_referencia,
    vento_referencia,
)
from modulo1_radiacao import saldo_radiacao
from modulo2_fluxo_calor_solo import fluxo_calor_solo, TAU_D_PADRAO
from modulo3_resistencia_aerodinamica import resistencia_aerodinamica
from modulo4_fluxo_momentum import fluxo_momentum
from modulo5_calor_sensivel import fluxo_calor_sensivel
from modulo6_calor_latente import fluxo_calor_latente, es_saturacao
from modulo7_balanco_hidrico import balanco_hidrico_solo


# ---------------------------------------------------------------
# 1) PARAMETROS DO MODELO (valores do slide "Hipoteses*", solo nu)
# ---------------------------------------------------------------
ALBEDO = 0.30          # alpha ~ 0,30
EMISSIVIDADE = 0.97    # epsilon ~ 0,97
CS = 1.0e6             # capacidade termica volumetrica (J K-1 m-3)
D_CAMADA = 0.10        # espessura da camada de solo prognostica (m)
TAU_D = TAU_D_PADRAO   # ~ 1 dia / 2*pi
Z0 = 0.01              # comprimento de rugosidade (m)
ZR = 10.0              # altura de referencia (m)
THETA_S = 0.50         # umidade de saturacao do solo
D_SOLO_HIDRICO = 0.5   # espessura considerada no balanco hidrico (m)
TD_PROFUNDO = 297.0    # temperatura do solo profundo (K), ~quase constante

DT = 300.0             # passo de tempo (s) = 5 min
N_DIAS_SPINUP = 4      # dias de "aquecimento" do modelo
N_DIAS_TOTAL = N_DIAS_SPINUP + 1   # ultimo dia e o que plotamos

PASSOS_POR_DIA = int(86400 / DT)
N_PASSOS = PASSOS_POR_DIA * N_DIAS_TOTAL


def rodar_modelo(theta_inicial=0.30, T_inicial=297.0):
    """
    Executa a integracao temporal completa e retorna um dicionario
    com as series temporais (apenas do ultimo dia) de todas as
    variaveis relevantes.
    """
    T = T_inicial
    theta = theta_inicial

    # listas para guardar o historico (todos os dias, depois recortamos)
    hist = {k: [] for k in [
        "hora_do_dia", "tempo_h",
        "SWd", "LWd", "Tr", "er", "Ur", "RH",
        "Rn", "G", "ra", "tau", "H", "LE",
        "T", "theta", "h_solo", "rsoil", "es_T",
    ]}

    for n in range(N_PASSOS):
        tempo_s = n * DT
        tempo_h = tempo_s / 3600.0
        hora_do_dia = tempo_h % 24.0

        # --------- MODULO 0: forcantes ---------
        SWd = radiacao_onda_curta_incidente(hora_do_dia)
        Tr = temperatura_referencia(hora_do_dia)
        LWd = radiacao_onda_longa_incidente(Tr)
        RH = umidade_relativa_referencia(hora_do_dia)
        Ur = vento_referencia(hora_do_dia)
        er = RH * es_saturacao(Tr)  # pressao de vapor no nivel de referencia

        # --------- MODULO 1: radiacao ---------
        Rn, SWu, LWu = saldo_radiacao(SWd, LWd, T, ALBEDO, EMISSIVIDADE)

        # --------- MODULO 2: calor no solo ---------
        G = fluxo_calor_solo(T, TD_PROFUNDO, CS, D_CAMADA, TAU_D)

        # --------- MODULO 3: resistencia aerodinamica ---------
        ra, CDN = resistencia_aerodinamica(Ur, ZR, Z0)

        # --------- MODULO 4: momentum (diagnostico) ---------
        tau = fluxo_momentum(Ur, ra)

        # --------- MODULO 5: calor sensivel ---------
        H = fluxo_calor_sensivel(T, Tr, ra)

        # --------- MODULO 6: calor latente ---------
        LE, extra = fluxo_calor_latente(T, er, ra, theta, THETA_S)

        # --------- Fechamento: dT/dt = (Rn - G - H - LE)/(Cs*D) ---------
        dTdt = (Rn - G - H - LE) / (CS * D_CAMADA)
        T_novo = T + dTdt * DT

        # --------- MODULO 7: balanco hidrico do solo ---------
        theta_novo, E_ms = balanco_hidrico_solo(theta, LE, DT, D_SOLO_HIDRICO)

        # guarda estado ANTES de avancar (estado "atual" ja calculado)
        hist["hora_do_dia"].append(hora_do_dia)
        hist["tempo_h"].append(tempo_h)
        hist["SWd"].append(SWd)
        hist["LWd"].append(LWd)
        hist["Tr"].append(Tr)
        hist["er"].append(er)
        hist["Ur"].append(Ur)
        hist["RH"].append(RH)
        hist["Rn"].append(Rn)
        hist["G"].append(G)
        hist["ra"].append(ra)
        hist["tau"].append(tau)
        hist["H"].append(H)
        hist["LE"].append(LE)
        hist["T"].append(T)
        hist["theta"].append(theta)
        hist["h_solo"].append(extra["h"])
        hist["rsoil"].append(extra["rsoil"])
        hist["es_T"].append(extra["es_T"])

        # avanca no tempo
        T = T_novo
        theta = theta_novo

    # converte para arrays numpy
    for k in hist:
        hist[k] = np.array(hist[k])

    # recorta apenas o ULTIMO dia (ciclo diurno ja estabilizado)
    idx_ultimo_dia = slice(PASSOS_POR_DIA * N_DIAS_SPINUP, N_PASSOS)
    ultimo_dia = {k: v[idx_ultimo_dia] for k, v in hist.items()}
    return ultimo_dia, hist


def salvar_csv(dados, caminho):
    colunas = list(dados.keys())
    n = len(dados[colunas[0]])
    with open(caminho, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(colunas)
        for i in range(n):
            w.writerow([f"{dados[c][i]:.4f}" for c in colunas])


def plotar_resultados(d, caminho_png):
    fig, axs = plt.subplots(4, 1, figsize=(9, 13), sharex=True)

    # Painel 1: forcantes de radiacao e temperatura do ar
    ax = axs[0]
    ax.plot(d["hora_do_dia"], d["SWd"], color="tab:orange", label="SWd (incidente)")
    ax.set_ylabel("Radiacao O.C. (W/m2)")
    ax2 = ax.twinx()
    ax2.plot(d["hora_do_dia"], d["Tr"] - 273.15, color="tab:red", label="Tr (ar)")
    ax2.plot(d["hora_do_dia"], d["T"] - 273.15, color="black", ls="--", label="T (superficie)")
    ax2.set_ylabel("Temperatura (C)")
    linhas = ax.get_lines() + ax2.get_lines()
    ax.legend(linhas, [l.get_label() for l in linhas], loc="upper left", fontsize=8)
    ax.set_title("Modulo 0/1  Forcante solar e temperaturas")

    # Painel 2: componentes do balanco de energia
    ax = axs[1]
    ax.plot(d["hora_do_dia"], d["Rn"], label="Rn", color="black")
    ax.plot(d["hora_do_dia"], d["G"], label="G", color="tab:brown")
    ax.plot(d["hora_do_dia"], d["H"], label="H", color="tab:red")
    ax.plot(d["hora_do_dia"], d["LE"], label="LE", color="tab:blue")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_ylabel("Fluxo (W/m2)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("Modulos 1-2-5-6  Balanco de energia: Rn = G + H + LE")

    # Painel 3: umidade do solo e resistencias
    ax = axs[2]
    ax.plot(d["hora_do_dia"], d["theta"], color="tab:blue", label=" (umidade do solo)")
    ax.set_ylabel(" (m3/m3)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(d["hora_do_dia"], d["rsoil"], color="tab:green", label="r_soil")
    ax2.plot(d["hora_do_dia"], d["ra"], color="tab:purple", label="r_a")
    ax2.set_ylabel("Resistencia (s/m)")
    linhas = ax.get_lines() + ax2.get_lines()
    ax.legend(linhas, [l.get_label() for l in linhas], loc="upper left", fontsize=8)
    ax.set_title("Modulos 3 e 7  Resistencias e balanco hidrico do solo")

    # Painel 4: momentum e vento
    ax = axs[3]
    ax.plot(d["hora_do_dia"], d["tau"], color="tab:purple", label="tau (momentum)")
    ax.set_ylabel("tau (Pa)")
    ax2 = ax.twinx()
    ax2.plot(d["hora_do_dia"], d["Ur"], color="tab:cyan", label="Ur (vento)")
    ax2.set_ylabel("Ur (m/s)")
    linhas = ax.get_lines() + ax2.get_lines()
    ax.legend(linhas, [l.get_label() for l in linhas], loc="upper left", fontsize=8)
    ax.set_xlabel("Hora local (h)")
    ax.set_title("Modulo 4  Fluxo de momentum")

    for ax in axs:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(caminho_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    ultimo_dia, historico_completo = rodar_modelo()

    salvar_csv(ultimo_dia, os.path.join(out_dir, "ciclo_diurno_ultimo_dia.csv"))
    plotar_resultados(ultimo_dia, os.path.join(out_dir, "ciclo_diurno_solo_nu.png"))

    # Resumo no terminal
    print("=== RESUMO DO ULTIMO DIA SIMULADO ===")
    print(f"T  min/max : {ultimo_dia['T'].min()-273.15:.2f} / {ultimo_dia['T'].max()-273.15:.2f} C")
    print(f"Tr min/max : {ultimo_dia['Tr'].min()-273.15:.2f} / {ultimo_dia['Tr'].max()-273.15:.2f} C")
    print(f"Rn min/max : {ultimo_dia['Rn'].min():.1f} / {ultimo_dia['Rn'].max():.1f} W/m2")
    print(f"H  min/max : {ultimo_dia['H'].min():.1f} / {ultimo_dia['H'].max():.1f} W/m2")
    print(f"LE min/max : {ultimo_dia['LE'].min():.1f} / {ultimo_dia['LE'].max():.1f} W/m2")
    print(f"G  min/max : {ultimo_dia['G'].min():.1f} / {ultimo_dia['G'].max():.1f} W/m2")
    print(f"theta ini/fim do dia: {ultimo_dia['theta'][0]:.3f} / {ultimo_dia['theta'][-1]:.3f}")
    print(f"\nArquivos salvos em: {out_dir}")
