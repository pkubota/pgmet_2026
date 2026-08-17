# -*- coding: utf-8 -*-
"""
MODELO CONCEITUAL DO REGIME DE TRANSICAO CONVECCAO RASA -> PROFUNDA
====================================================================
Curso: Conveccao Atmosferica - Regimes de Transicao
Autor: material didatico gerado para uso em sala de aula

OBJETIVO
--------
Este e um "toy model" (modelo conceitual simplificado, NAO um CRM/SCM
operacional) que integra em UM unico codigo os processos fisicos
centrais discutidos no curso:

  1. Crescimento da Camada Limite Convectiva (CLC) por aquecimento de
     superficie (modelo de encroachment / "slab model");
  2. Balanco de umidade da CLC com entranhamento de ar da troposfera
     livre no topo;
  3. Conveccao rasa: fluxo de massa de base de nuvem (cloud-base mass
     flux) ligado ao aquecimento de superficie;
  4. Umedecimento da troposfera livre baixa por DETRANHAMENTO da
     conveccao rasa (o processo-chave da transicao, seguindo o
     mecanismo discutido em Kuang & Bretherton 2006; Khairoutdinov &
     Randall 2006; Zhang & Klein 2010);
  5. Erosao progressiva do CIN (inibicao convectiva) a medida que a
     troposfera livre umedece -> ligacao direta com a FUNCAO DE
     GATILHO (trigger) discutida nos slides 23-24 e 166 do material
     do curso;
  6. Crescimento do CAPE ao longo do dia;
  7. Criterio de disparo (trigger) da conveccao profunda: CIN cai
     abaixo de um limiar critico E o CAPE excede um minimo.

Este modelo e DIDATICO: as constantes de fechamento (c1, tau_relax,
q_scale, etc.) foram escolhidas para reproduzir o comportamento
QUALITATIVO observado (ciclo diurno conveccao rasa -> profunda, pico
de precipitacao profunda no inicio-meio da tarde), nao para
reproduzir quantitativamente um caso real. Ele serve como uma
"bancada de testes" conceitual: mude os parametros e veja como o
horario de transicao se desloca.

USO EM SALA
-----------
- Rode o script e mostre a Figura 1 (diagrama esquematico) e a
  Figura 2 (ciclo diurno simulado) juntas.
- Proponha exercicios: e.g., "o que acontece com o horario de
  transicao se tau_relax (subsidencia de grande escala) diminuir?"
  (resposta esperada: transicao atrasa ou nao ocorre - regime mais
  seco / oceanico) ou "o que acontece se aumentarmos SHF_max?"
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# (fix acima: evita UnicodeEncodeError em terminais Windows que nao usam UTF-8
#  por padrao -- necessario para os acentos e simbolos gregos usados nos prints)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 1. PARAMETROS FISICOS E CONSTANTES
# =====================================================================
g = 9.81          # gravidade [m/s2]
cp = 1004.0        # calor especifico do ar seco [J/kg/K]
Lv = 2.5e6         # calor latente de vaporizacao [J/kg]
rho = 1.15         # densidade do ar proximo a superficie [kg/m3]

# --- Perfil ambiental (troposfera livre) ---
gamma_theta = 5.0e-3   # taxa de variacao vertical de theta ambiente [K/m] (estavel)
q_ft0 = 4.0e-3         # umidade especifica "de fundo" da troposfera livre baixa [kg/kg] (seca, subsidente)
q_ft_sat = 12.0e-3     # limite superior de umedecimento pela conveccao rasa [kg/kg]

# --- Forcantes de superficie: ciclo diurno senoidal (06h-18h de sol) ---
t_sunrise = 6 * 3600.0     # nascer do sol [s a partir de 00h local]
daylength = 12 * 3600.0    # duracao do periodo com fluxos positivos [s]
SHF_max = 250.0            # fluxo de calor sensivel maximo [W/m2]
LHF_max = 150.0            # fluxo de calor latente maximo [W/m2]

def SHF(t):
    """Fluxo de calor sensivel de superficie (W/m2), ciclo diurno senoidal."""
    h = t - t_sunrise
    if 0.0 <= h <= daylength:
        return SHF_max * np.sin(np.pi * h / daylength)
    return 0.0

def LHF(t):
    """Fluxo de calor latente de superficie (W/m2), ciclo diurno senoidal."""
    h = t - t_sunrise
    if 0.0 <= h <= daylength:
        return LHF_max * np.sin(np.pi * h / daylength)
    return 0.0

# --- Parametros de fechamento (didaticos) ---
CIN0 = 120.0        # CIN inicial ao amanhecer [J/kg] (atmosfera ainda seca/inibida)
CIN_crit = 15.0      # limiar de CIN abaixo do qual o gatilho e considerado "vencivel" [J/kg]
CAPE_max = 1800.0    # CAPE potencial maximo (parcela nao diluida) [J/kg]
CAPE_crit = 800.0    # CAPE minimo exigido, junto com CIN baixo, para disparar conveccao profunda [J/kg]
q_scale = 3.0e-3     # escala de umedecimento que controla a erosao do CIN [kg/kg]
c1 = 1.6e-3          # eficiencia de conversao SHF -> fluxo de massa de base de nuvem rasa [kg/m2/s por W/m2]
tau_relax = 6 * 3600.0  # escala de tempo de subsidencia/mistura que seca a troposfera livre [s]
z_shallow = 20.0     # espessura de referencia da camada de detranhamento raso [m]

# =====================================================================
# 2. INTEGRACAO TEMPORAL (Euler explicito - fins didaticos)
# =====================================================================
dt = 60.0                       # passo de tempo [s]
t_end = 18 * 3600.0             # integra ate 18h local
times = np.arange(0.0, t_end, dt)
n = len(times)

h     = np.zeros(n)   # profundidade da Camada Limite Convectiva [m]
q_ml  = np.zeros(n)   # umidade especifica da CLC [kg/kg]
q_ft  = np.zeros(n)   # umidade especifica da troposfera livre baixa [kg/kg]
Mb    = np.zeros(n)   # fluxo de massa de base de nuvem rasa [kg/m2/s]
CIN   = np.zeros(n)   # inibicao convectiva [J/kg]
CAPE  = np.zeros(n)   # energia potencial convectiva disponivel [J/kg]

h[0]    = 100.0
q_ml[0] = 10.0e-3
q_ft[0] = q_ft0
CIN[0]  = CIN0
CAPE[0] = 0.0

trigger_time = None  # horario (s) em que a conveccao profunda e disparada

for i in range(1, n):
    t = times[i]
    shf = SHF(t)
    lhf = LHF(t)

    # --- (1) Crescimento da CLC: modelo de encroachment ---
    # d(h^2)/dt = 2*SHF / (rho*cp*gamma_theta)  =>  integrado em Euler:
    h2 = h[i-1]**2 + 2.0 * dt * max(shf, 0.0) / (rho * cp * gamma_theta)
    h[i] = np.sqrt(max(h2, 100.0**2))
    dh = h[i] - h[i-1]

    # --- (2) Balanco de umidade da CLC: fonte de superficie + entranhamento no topo ---
    q_ml[i] = q_ml[i-1] + dt * (max(lhf, 0.0) / (rho * Lv * h[i])) \
              + (q_ft[i-1] - q_ml[i-1]) * (dh / h[i])
    q_ml[i] = max(q_ml[i], 1.0e-4)

    # --- (3) Conveccao rasa: fluxo de massa de base de nuvem ---
    Mb[i] = c1 * max(shf, 0.0)

    # --- (4) Umedecimento da troposfera livre por DETRANHAMENTO raso,
    #         competindo com relaxamento por subsidencia de grande escala ---
    umedecimento = Mb[i] / rho * (q_ml[i] - q_ft[i-1]) / z_shallow
    secagem      = (q_ft[i-1] - q_ft0) / tau_relax
    q_ft[i] = np.clip(q_ft[i-1] + dt * (umedecimento - secagem), q_ft0, q_ft_sat)

    # --- (5) Erosao do CIN pelo umedecimento da troposfera livre ---
    # Quanto mais umida a troposfera livre, menor a diluicao por
    # entranhamento das plumas -> menos "gasto" de flutuabilidade -> CIN cai.
    CIN[i] = CIN0 * np.exp(-(q_ft[i] - q_ft0) / q_scale)

    # --- (6) Crescimento do CAPE (aquecimento + umedecimento da CLC) ---
    CAPE[i] = CAPE_max * (1.0 - np.exp(-(h[i] - 100.0) / 1200.0)) * (q_ml[i] / q_ml[0])

    # --- (7) Criterio de disparo (trigger) da conveccao profunda ---
    if trigger_time is None and CIN[i] < CIN_crit and CAPE[i] > CAPE_crit:
        trigger_time = t

# =====================================================================
# 3. DIAGNOSTICO E SAIDA
# =====================================================================
horas = times / 3600.0

if trigger_time is not None:
    print(f"Transicao rasa -> profunda disparada as {trigger_time/3600.0:.2f} h local")
else:
    print("Transicao NAO ocorreu no periodo simulado (CIN nunca caiu abaixo do limiar "
          "com CAPE suficiente). Regime permaneceu raso -> tente aumentar tau_relax "
          "ou SHF_max para representar um dia mais favoravel.")

# =====================================================================
# 4. FIGURA: CICLO DIURNO SIMULADO (para uso em aula)
# =====================================================================
fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
fig.suptitle("Modelo conceitual do regime de transicao rasa -> profunda", fontsize=15, fontweight="bold")

axs[0,0].plot(horas, [SHF(t) for t in times], color="firebrick", label="SHF")
axs[0,0].plot(horas, [LHF(t) for t in times], color="steelblue", label="LHF")
axs[0,0].set_ylabel("W/m^2")
axs[0,0].set_title("Fluxos de superficie")
axs[0,0].legend(fontsize=8)

axs[0,1].plot(horas, h, color="darkorange")
axs[0,1].set_ylabel("m")
axs[0,1].set_title("Profundidade da Camada Limite Convectiva (h)")

axs[1,0].plot(horas, q_ml*1000, color="seagreen", label="q CLC")
axs[1,0].plot(horas, q_ft*1000, color="teal", label="q troposfera livre")
axs[1,0].set_ylabel("g/kg")
axs[1,0].set_title("Umedecimento por detranhamento raso")
axs[1,0].legend(fontsize=8)

axs[1,1].plot(horas, Mb*1000, color="purple")
axs[1,1].set_ylabel("g/m^2/s")
axs[1,1].set_title("Fluxo de massa de base de nuvem rasa (Mb)")

axs[2,0].plot(horas, CIN, color="crimson", label="CIN")
axs[2,0].axhline(CIN_crit, color="crimson", ls="--", lw=1, label="CIN critico (gatilho)")
axs[2,0].set_ylabel("J/kg")
axs[2,0].set_xlabel("Hora local")
axs[2,0].set_title("Erosao do CIN (inibicao convectiva)")
axs[2,0].legend(fontsize=8)

axs[2,1].plot(horas, CAPE, color="navy", label="CAPE")
axs[2,1].axhline(CAPE_crit, color="navy", ls="--", lw=1, label="CAPE minimo p/ disparo")
axs[2,1].set_ylabel("J/kg")
axs[2,1].set_xlabel("Hora local")
axs[2,1].set_title("Crescimento do CAPE")
axs[2,1].legend(fontsize=8)

for ax in axs.flat:
    ax.grid(alpha=0.3)
    if trigger_time is not None:
        ax.axvline(trigger_time/3600.0, color="black", ls=":", lw=1.5)

if trigger_time is not None:
    axs[0,0].annotate("Transicao\nrasa->profunda", xy=(trigger_time/3600.0, SHF_max*0.9),
                       fontsize=8, ha="center", fontweight="bold")

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("ciclo_diurno_transicao.png", dpi=160)
print("Figura salva: ciclo_diurno_transicao.png")
