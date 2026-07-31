# -*- coding: utf-8 -*-
"""
simulacao_passo1.py
====================

Script principal do PASSO 1: cria uma coluna atmosferica, insere uma
camada de nuvem ja formada (agua de nuvem qc, Nc), integra os processos
de chuva quente por um periodo de tempo e plota:

    (a) Perfis verticais de qc, qr em diferentes instantes de tempo
    (b) Series temporais de qc, qr, Nc, Nr integrados na coluna
    (c) Precipitacao acumulada na superficie

Para rodar:  python3 simulacao_passo1.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microfisica.coluna_step1 import ColunaChuvaQuente

# ----------------------------------------------------------------------
# 1) Configuracao da coluna e da nuvem inicial
# ----------------------------------------------------------------------
coluna = ColunaChuvaQuente(nz=40, dz=100.0, T_base=293.0, p_base=95000.0)

# Insere uma camada de nuvem entre 1000 m e 2500 m de altura
k_base = int(1000 / coluna.dz)
k_topo = int(2500 / coluna.dz)
coluna.inserir_nuvem(k_base, k_topo, qc_valor=1.5e-3, Nc_valor=1.0e8)

print(f"Nuvem inserida entre {coluna.z[k_base]:.0f} m e {coluna.z[k_topo]:.0f} m")
print(f"qc inicial na nuvem: {1.5e-3*1000:.2f} g/kg | Nc inicial: 1.0e8 kg^-1 (~100 gotas/cm^3)")

# ----------------------------------------------------------------------
# 2) Integracao no tempo
# ----------------------------------------------------------------------
TEMPO_TOTAL = 3600.0  # 1 hora de simulacao
DT = 5.0               # passo de tempo (s)
historico = coluna.integrar(TEMPO_TOTAL, dt=DT, salvar_a_cada=120.0)

print(f"\nPrecipitacao acumulada na superficie ao final de {TEMPO_TOTAL/60:.0f} min: "
      f"{coluna.precip_superficie_mm:.4f} mm")

# ----------------------------------------------------------------------
# 3) Grafico (a): perfis verticais de qc e qr em diferentes tempos
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
indices_plot = np.linspace(0, len(historico["t"]) - 1, 6).astype(int)
cores = plt.cm.viridis(np.linspace(0, 1, len(indices_plot)))

for i, idx in enumerate(indices_plot):
    t_min = historico["t"][idx] / 60.0
    axes[0].plot(np.array(historico["qc"][idx]) * 1000, coluna.z,
                 color=cores[i], label=f"t={t_min:.0f} min")
    axes[1].plot(np.array(historico["qr"][idx]) * 1000, coluna.z,
                 color=cores[i], label=f"t={t_min:.0f} min")

axes[0].set_xlabel("qc (g/kg)")
axes[0].set_ylabel("Altura (m)")
axes[0].set_title("Agua de nuvem (qc)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("qr (g/kg)")
axes[1].set_title("Agua de chuva (qr)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.suptitle("Passo 1 - Evolucao vertical de qc e qr (esquema de chuva quente)")
plt.tight_layout()
plt.savefig("/home/claude/microfisica_project/fig_perfis_verticais.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------
# 4) Grafico (b): series temporais integradas na coluna (massa total)
# ----------------------------------------------------------------------
qc_total = [np.sum(np.array(qc) * coluna.rho) * coluna.dz for qc in historico["qc"]]
qr_total = [np.sum(np.array(qr) * coluna.rho) * coluna.dz for qr in historico["qr"]]
Nc_total = [np.sum(np.array(nc) * coluna.rho) * coluna.dz for nc in historico["Nc"]]
Nr_total = [np.sum(np.array(nr) * coluna.rho) * coluna.dz for nr in historico["Nr"]]
t_min = np.array(historico["t"]) / 60.0

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
axes[0].plot(t_min, qc_total, label="Agua de nuvem (qc)", color="tab:blue")
axes[0].plot(t_min, qr_total, label="Agua de chuva (qr)", color="tab:orange")
axes[0].set_ylabel("Massa total na coluna (kg/m$^2$)")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_title("Massa de agua condensada na coluna")

axes[1].plot(t_min, Nc_total, label="N total de goticulas (Nc)", color="tab:blue")
axes[1].plot(t_min, Nr_total, label="N total de gotas de chuva (Nr)", color="tab:orange")
axes[1].set_ylabel("N total na coluna (# / m$^2$)")
axes[1].set_xlabel("Tempo (min)")
axes[1].set_yscale("log")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_title("Concentracao numerica total (nota a escala log)")

plt.tight_layout()
plt.savefig("/home/claude/microfisica_project/fig_series_temporais.png", dpi=130)
plt.close()

print("\nFiguras salvas:")
print(" - fig_perfis_verticais.png")
print(" - fig_series_temporais.png")
