# -*- coding: utf-8 -*-
"""
MODELO DE SUPERFICIE - SOLO NU - MULTICAMADAS (DIFUSAO)
====================================================================
Versao estendida do modelo diurno, agora com PERFIS de temperatura e
umidade no solo (em vez de uma unica camada com force-restore),
seguindo os slides "Modelo com Varias Camadas de Solo":

    0. Forcantes atmosfericas          (modulo0)
    1. Saldo de radiacao               (modulo1)  -> usa T da superficie
    3. Resistencia aerodinamica        (modulo3)
    4. Fluxo de momentum               (modulo4)
    5. Fluxo de calor sensivel         (modulo5)  -> usa T da superficie
    6. Fluxo de calor latente          (modulo6)  -> usa T da superficie e theta[0]
    8. Difusao de umidade do solo      (modulo8)  <-- NOVO
    9. Difusao de calor no solo        (modulo9)  <-- NOVO

Diferenca-chave em relacao ao modelo de 1 camada (main_modelo_diurno.py):
o fluxo de calor no solo G NAO e mais parametrizado por force-restore;
ele emerge NATURALMENTE da equacao de difusao (modulo9), que precisa
de uma temperatura de superficie T_s como condicao de contorno.

T_s e obtida a cada passo de tempo resolvendo o BALANCO DE ENERGIA DA
SUPERFICIE (conforme o slide "Modelo de temperatura do solo"):

    (1-alpha)*SWd + LWd - epsilon*sigma*Ts^4 = G(Ts) + H(Ts) + LE(Ts)

onde G(Ts) = K_T_interf(0) * (Ts - T[0]) / dz_tilde(0)  (fluxo difusivo
para a primeira camada), H(Ts) e LE(Ts) como nos modulos 5 e 6.

Essa equacao e nao linear em Ts (por causa de Ts^4 e de es(Ts)) e e
resolvida por um metodo de NEWTON simples (poucas iteracoes, derivada
numerica), a cada passo de tempo.
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
from modulo3_resistencia_aerodinamica import resistencia_aerodinamica
from modulo4_fluxo_momentum import fluxo_momentum
from modulo5_calor_sensivel import fluxo_calor_sensivel
from modulo6_calor_latente import fluxo_calor_latente, es_saturacao
from modulo8_difusao_umidade_solo import atualizar_umidade_solo, obter_parametros_solo
from modulo9_difusao_calor_solo import (
    difusao_calor_solo_implicito,
    condutividade_termica,
    capacidade_termica,
)


# ---------------------------------------------------------------
# 1) CONFIGURACAO DAS CAMADAS DE SOLO (multicamadas)
# ---------------------------------------------------------------
DZ = np.array([0.05, 0.10, 0.15, 0.25, 0.35, 0.50])  # espessuras (m)
# camada 0 = 5 cm, seguindo a recomendacao do slide "O modelo
# termodinamico de duas camadas de solo": camada fina no topo para
# capturar o gradiente termico diurno; camadas mais espessas abaixo
# para representar o armazenamento de calor/variacoes mais lentas.
N_CAMADAS = len(DZ)
PROF_TOPO = np.concatenate(([0.0], np.cumsum(DZ)[:-1]))
PROF_CENTRO = PROF_TOPO + DZ / 2.0
PROF_FUNDO_TOTAL = np.sum(DZ)

CLASSE_SOLO = "franco_argiloso"
ALBEDO = 0.30
EMISSIVIDADE = 0.97
Z0 = 0.01
ZR = 10.0
T_FUNDO = 297.0   # K, "temperatura constante especificada Td_bot" (slide)

DT = 300.0                    # passo de tempo (s)
N_DIAS_SPINUP = 4
N_DIAS_TOTAL = N_DIAS_SPINUP + 1
PASSOS_POR_DIA = int(86400 / DT)
N_PASSOS = PASSOS_POR_DIA * N_DIAS_TOTAL

L_VAPORIZACAO = 2.5e6
RHO_AGUA = 1000.0


def balanco_energia_superficie(Ts, SWd, LWd, Tr, er, ra, T0, theta0,
                                K_T0, dz_tilde0, theta_s):
    """
    Residuo do balanco de energia da superficie:
        Rn(Ts) - G(Ts) - H(Ts) - LE(Ts) = 0
    Retorna o residuo e os termos individuais (para diagnostico).
    """
    Rn, SWu, LWu = saldo_radiacao(SWd, LWd, Ts, ALBEDO, EMISSIVIDADE)
    G = K_T0 * (Ts - T0) / dz_tilde0
    H = fluxo_calor_sensivel(Ts, Tr, ra)
    LE, extra = fluxo_calor_latente(Ts, er, ra, theta0, theta_s)
    residuo = Rn - G - H - LE
    return residuo, Rn, G, H, LE, extra


def resolver_temperatura_superficie(Ts_guess, SWd, LWd, Tr, er, ra, T0,
                                     theta0, K_T0, dz_tilde0, theta_s,
                                     n_iter=8, eps=0.05):
    """
    Metodo de Newton (derivada numerica) para achar Ts que zera o
    residuo do balanco de energia da superficie.
    """
    Ts = Ts_guess
    for _ in range(n_iter):
        f0, *_ = balanco_energia_superficie(
            Ts, SWd, LWd, Tr, er, ra, T0, theta0, K_T0, dz_tilde0, theta_s)
        f1, *_ = balanco_energia_superficie(
            Ts + eps, SWd, LWd, Tr, er, ra, T0, theta0, K_T0, dz_tilde0, theta_s)
        dfdT = (f1 - f0) / eps
        if abs(dfdT) < 1.0e-8:
            break
        passo = f0 / dfdT
        Ts = Ts - passo
        Ts = float(np.clip(Ts, 250.0, 340.0))
        if abs(passo) < 1.0e-4:
            break
    return Ts


def rodar_modelo(T_inicial=297.0, theta_inicial=0.30):
    psi_s, theta_s, b, Ks = obter_parametros_solo(CLASSE_SOLO)

    T = np.full(N_CAMADAS, T_inicial)
    theta = np.full(N_CAMADAS, theta_inicial)
    Ts_guess = T_inicial

    hist = {k: [] for k in [
        "hora_do_dia", "tempo_h",
        "SWd", "LWd", "Tr", "er", "Ur", "RH",
        "Ts", "Rn", "G", "H", "LE", "tau", "ra",
    ]}
    hist_T_perfil = []
    hist_theta_perfil = []

    for n in range(N_PASSOS):
        tempo_h = n * DT / 3600.0
        hora_do_dia = tempo_h % 24.0

        # --------- MODULO 0: forcantes ---------
        SWd = radiacao_onda_curta_incidente(hora_do_dia)
        Tr = temperatura_referencia(hora_do_dia)
        LWd = radiacao_onda_longa_incidente(Tr)
        RH = umidade_relativa_referencia(hora_do_dia)
        Ur = vento_referencia(hora_do_dia)
        er = RH * es_saturacao(Tr)

        # --------- MODULO 3: resistencia aerodinamica ---------
        ra, CDN = resistencia_aerodinamica(Ur, ZR, Z0)

        # --------- Propriedades termicas da camada 0 (p/ contorno) ---------
        K_T_perfil = condutividade_termica(theta, psi_s, theta_s, b)
        dz_tilde0 = 0.5 * DZ[0]

        # --------- Resolve Ts (balanco de energia da superficie) ---------
        Ts = resolver_temperatura_superficie(
            Ts_guess, SWd, LWd, Tr, er, ra, T[0], theta[0],
            K_T_perfil[0], dz_tilde0, theta_s)
        Ts_guess = Ts  # usa como chute inicial no proximo passo

        residuo, Rn, G, H, LE, extra = balanco_energia_superficie(
            Ts, SWd, LWd, Tr, er, ra, T[0], theta[0],
            K_T_perfil[0], dz_tilde0, theta_s)

        # --------- MODULO 4: momentum (diagnostico) ---------
        tau = fluxo_momentum(Ur, ra)

        # --------- MODULO 9: difusao de calor no solo ---------
        T_novo, K_T_perfil, C_perfil = difusao_calor_solo_implicito(
            T, theta, DZ, DT, Ts, T_FUNDO, CLASSE_SOLO)

        # --------- MODULO 8: difusao de umidade do solo ---------
        E_dir = LE / (L_VAPORIZACAO * RHO_AGUA)  # m/s (LE>0 => evaporacao)
        theta_novo = atualizar_umidade_solo(
            theta, DZ, DT, CLASSE_SOLO, evaporacao=E_dir, precipitacao=0.0)

        # --------- guarda historico ---------
        hist["hora_do_dia"].append(hora_do_dia)
        hist["tempo_h"].append(tempo_h)
        hist["SWd"].append(SWd)
        hist["LWd"].append(LWd)
        hist["Tr"].append(Tr)
        hist["er"].append(er)
        hist["Ur"].append(Ur)
        hist["RH"].append(RH)
        hist["Ts"].append(Ts)
        hist["Rn"].append(Rn)
        hist["G"].append(G)
        hist["H"].append(H)
        hist["LE"].append(LE)
        hist["tau"].append(tau)
        hist["ra"].append(ra)
        hist_T_perfil.append(T.copy())
        hist_theta_perfil.append(theta.copy())

        T = T_novo
        theta = theta_novo

    for k in hist:
        hist[k] = np.array(hist[k])
    hist_T_perfil = np.array(hist_T_perfil)        # (N_PASSOS, N_CAMADAS)
    hist_theta_perfil = np.array(hist_theta_perfil)

    idx_ultimo = slice(PASSOS_POR_DIA * N_DIAS_SPINUP, N_PASSOS)
    ultimo_dia = {k: v[idx_ultimo] for k, v in hist.items()}
    ultimo_dia["T_perfil"] = hist_T_perfil[idx_ultimo]
    ultimo_dia["theta_perfil"] = hist_theta_perfil[idx_ultimo]

    return ultimo_dia


def salvar_csv(dados, caminho):
    colunas = [k for k in dados.keys() if k not in ("T_perfil", "theta_perfil")]
    n = len(dados[colunas[0]])
    with open(caminho, "w", newline="") as f:
        w = csv.writer(f)
        header = colunas + [f"T_camada{i}_m{PROF_CENTRO[i]:.2f}" for i in range(N_CAMADAS)] \
                          + [f"theta_camada{i}_m{PROF_CENTRO[i]:.2f}" for i in range(N_CAMADAS)]
        w.writerow(header)
        for i in range(n):
            linha = [f"{dados[c][i]:.4f}" for c in colunas]
            linha += [f"{dados['T_perfil'][i, j]:.4f}" for j in range(N_CAMADAS)]
            linha += [f"{dados['theta_perfil'][i, j]:.4f}" for j in range(N_CAMADAS)]
            w.writerow(linha)


def plotar_resultados(d, caminho_png):
    fig, axs = plt.subplots(4, 1, figsize=(9, 14))

    # Painel 1: forcantes e Ts
    ax = axs[0]
    ax.plot(d["hora_do_dia"], d["SWd"], color="tab:orange", label="SWd")
    ax.set_ylabel("SWd (W/m2)")
    ax2 = ax.twinx()
    ax2.plot(d["hora_do_dia"], d["Tr"] - 273.15, color="tab:red", label="Tr (ar)")
    ax2.plot(d["hora_do_dia"], d["Ts"] - 273.15, color="black", ls="--", label="Ts (superficie)")
    ax2.plot(d["hora_do_dia"], d["T_perfil"][:, -1] - 273.15, color="tab:brown",
              ls=":", label=f"T camada mais funda ({PROF_CENTRO[-1]:.2f} m)")
    ax2.set_ylabel("Temperatura (C)")
    linhas = ax.get_lines() + ax2.get_lines()
    ax.legend(linhas, [l.get_label() for l in linhas], loc="upper left", fontsize=8)
    ax.set_title("Forcante solar e temperaturas (superficie via balanco de energia)")
    ax.set_xlim(0, 24)
    ax.grid(alpha=0.3)

    # Painel 2: balanco de energia
    ax = axs[1]
    ax.plot(d["hora_do_dia"], d["Rn"], label="Rn", color="black")
    ax.plot(d["hora_do_dia"], d["G"], label="G (difusivo)", color="tab:brown")
    ax.plot(d["hora_do_dia"], d["H"], label="H", color="tab:red")
    ax.plot(d["hora_do_dia"], d["LE"], label="LE", color="tab:blue")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_ylabel("Fluxo (W/m2)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("Balanco de energia da superficie: Rn = G + H + LE")
    ax.set_xlim(0, 24)
    ax.grid(alpha=0.3)

    # Painel 3: perfil de temperatura no tempo (profundidade x hora)
    ax = axs[2]
    prof_bordas = np.concatenate(([0.0], np.cumsum(DZ)))          # 8 bordas p/ 7 camadas
    malha_hora = np.concatenate((d["hora_do_dia"], [24.0]))       # 289 bordas p/ 288 passos
    T_plot = d["T_perfil"] - 273.15                                # (288, 7)
    pcm = ax.pcolormesh(malha_hora, prof_bordas, T_plot.T, shading="flat", cmap="turbo")
    ax.invert_yaxis()
    ax.set_ylabel("Profundidade (m)")
    ax.set_title("Perfil de temperatura do solo T(z,t)  [modulo 9 - difusao]")
    fig.colorbar(pcm, ax=ax, label="T (C)")
    ax.set_xlim(0, 24)

    # Painel 4: perfil de umidade no tempo (profundidade x hora)
    ax = axs[3]
    theta_plot = d["theta_perfil"]                                 # (288, 7)
    pcm2 = ax.pcolormesh(malha_hora, prof_bordas, theta_plot.T, shading="flat", cmap="Blues")
    ax.invert_yaxis()
    ax.set_ylabel("Profundidade (m)")
    ax.set_xlabel("Hora local (h)")
    ax.set_title("Perfil de umidade do solo theta(z,t)  [modulo 8 - difusao]")
    fig.colorbar(pcm2, ax=ax, label="theta (m3/m3)")
    ax.set_xlim(0, 24)

    fig.tight_layout()
    fig.savefig(caminho_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    ultimo_dia = rodar_modelo()

    salvar_csv(ultimo_dia, os.path.join(out_dir, "ciclo_diurno_multicamadas.csv"))
    plotar_resultados(ultimo_dia, os.path.join(out_dir, "ciclo_diurno_multicamadas.png"))

    print("=== RESUMO DO ULTIMO DIA SIMULADO (modelo multicamadas) ===")
    print(f"Ts min/max        : {ultimo_dia['Ts'].min()-273.15:.2f} / {ultimo_dia['Ts'].max()-273.15:.2f} C")
    print(f"Tr min/max        : {ultimo_dia['Tr'].min()-273.15:.2f} / {ultimo_dia['Tr'].max()-273.15:.2f} C")
    print(f"T camada 0 min/max: {ultimo_dia['T_perfil'][:,0].min()-273.15:.2f} / {ultimo_dia['T_perfil'][:,0].max()-273.15:.2f} C")
    print(f"T camada -1 (fundo, {PROF_CENTRO[-1]:.2f} m) min/max: "
          f"{ultimo_dia['T_perfil'][:,-1].min()-273.15:.2f} / {ultimo_dia['T_perfil'][:,-1].max()-273.15:.2f} C")
    print(f"Rn min/max        : {ultimo_dia['Rn'].min():.1f} / {ultimo_dia['Rn'].max():.1f} W/m2")
    print(f"G  min/max        : {ultimo_dia['G'].min():.1f} / {ultimo_dia['G'].max():.1f} W/m2")
    print(f"H  min/max        : {ultimo_dia['H'].min():.1f} / {ultimo_dia['H'].max():.1f} W/m2")
    print(f"LE min/max        : {ultimo_dia['LE'].min():.1f} / {ultimo_dia['LE'].max():.1f} W/m2")
    print(f"theta camada0 ini/fim: {ultimo_dia['theta_perfil'][0,0]:.3f} / {ultimo_dia['theta_perfil'][-1,0]:.3f}")
    print(f"\nArquivos salvos em: {out_dir}")
