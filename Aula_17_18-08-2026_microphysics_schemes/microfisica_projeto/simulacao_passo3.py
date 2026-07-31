# -*- coding: utf-8 -*-
"""
simulacao_passo3.py
=====================

Script principal do PASSO 3: mesma configuracao do Passo 2 (nuvem
profunda cruzando 0 graus C), mas agora com neve e graupel disponiveis, e
uma pequena quantidade de chuva pre-existente numa camada fria para
exercitar o congelamento heterogeneo de chuva (Pgfzr).

Gera:
    (a) Perfis verticais de qi, qs, qg em diferentes instantes
    (b) Serie temporal de todas as categorias de agua na coluna
    (c) Perfil final de temperatura com os niveis de 0 graus C, -5 graus C e -8/-3 graus C
        (faixa de Hallett-Mossop) marcados

Para rodar:  python3 simulacao_passo3.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microfisica.coluna_step3 import ColunaFaseMista
from microfisica.constantes import T0

# ----------------------------------------------------------------------
# 1) Configuracao da coluna
# ----------------------------------------------------------------------
coluna = ColunaFaseMista(nz=80, dz=100.0, T_base=293.0, p_base=95000.0)

k_base = int(1500 / coluna.dz)
k_topo = int(6000 / coluna.dz)
coluna.inserir_nuvem(k_base, k_topo, qc_valor=1.0e-3, Nc_valor=2.0e8)

k_fria = int(4500 / coluna.dz)
coluna.qr[k_fria] = 5.0e-4
coluna.Nr[k_fria] = 5.0e5

k_zero_C = int(np.argmin(np.abs(coluna.T - T0)))
print(f"Nuvem inserida entre {coluna.z[k_base]:.0f} m e {coluna.z[k_topo]:.0f} m")
print(f"Nivel de 0 graus C:  {coluna.z[k_zero_C]:.0f} m")
print(f"Chuva pre-existente inserida em {coluna.z[k_fria]:.0f} m "
      f"(T={coluna.T[k_fria]-273.15:.1f} graus C) para testar Pgfzr")

# ----------------------------------------------------------------------
# 2) Integracao no tempo
# ----------------------------------------------------------------------
TEMPO_TOTAL = 1800.0
DT = 2.0
historico = coluna.integrar(TEMPO_TOTAL, dt=DT, salvar_a_cada=60.0)

print(f"\nPrecipitacao acumulada na superficie: {coluna.precip_superficie_mm:.4f} mm")

# ----------------------------------------------------------------------
# 3) Grafico (a): perfis verticais finais de todas as categorias de gelo
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
indices_plot = np.linspace(0, len(historico["t"]) - 1, 6).astype(int)
cores = plt.cm.viridis(np.linspace(0, 1, len(indices_plot)))

campos = ["qi", "qs", "qg", "qr"]
titulos = ["Gelo de nuvem (qi)", "Neve (qs)", "Graupel (qg)", "Chuva (qr)"]

for ax, campo, titulo in zip(axes, campos, titulos):
    for i, idx in enumerate(indices_plot):
        t_min = historico["t"][idx] / 60.0
        ax.plot(np.array(historico[campo][idx]) * 1000, coluna.z,
                 color=cores[i], label=f"t={t_min:.0f} min" if campo == "qi" else None)
    ax.axhline(coluna.z[k_zero_C], color="gray", ls="--", lw=1)
    ax.set_xlabel(f"{campo} (g/kg)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Altura (m)")
axes[0].legend(fontsize=7)

plt.suptitle("Passo 3 - Neve, graupel e interacoes de fase mista")
plt.tight_layout()
plt.savefig("/home/claude/microfisica_project/fig_passo3_perfis.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------
# 4) Grafico (b): series temporais de todas as categorias
# ----------------------------------------------------------------------
def massa_total(campo):
    return [np.sum(np.array(v) * coluna.rho) * coluna.dz for v in historico[campo]]

t_min = np.array(historico["t"]) / 60.0

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(t_min, massa_total("qc"), label="Agua de nuvem (qc)", color="tab:blue")
ax.plot(t_min, massa_total("qi"), label="Gelo de nuvem (qi)", color="tab:cyan")
ax.plot(t_min, massa_total("qs"), label="Neve (qs)", color="tab:purple")
ax.plot(t_min, massa_total("qg"), label="Graupel (qg)", color="tab:gray")
ax.plot(t_min, massa_total("qr"), label="Chuva (qr)", color="tab:orange")
ax.set_xlabel("Tempo (min)")
ax.set_ylabel("Massa total na coluna (kg/m$^2$)")
ax.set_title("Evolucao de todas as categorias de agua (Passo 3)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/microfisica_project/fig_passo3_series.png", dpi=130)
plt.close()

print("\nFiguras salvas:")
print(" - fig_passo3_perfis.png")
print(" - fig_passo3_series.png")
