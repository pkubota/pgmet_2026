# -*- coding: utf-8 -*-
"""
simulacao_passo2.py
=====================

Script principal do PASSO 2: cria uma coluna atmosferica com uma
camada de nuvem que se estende da fase QUENTE (base) ate a fase FRIA
(topo, T<0 graus C), permitindo observar o efeito Wegener-Bergeron-Findeisen
(WBF) -- crescimento do gelo as custas da agua liquida na parte fria
da nuvem -- alem de todos os processos ja vistos no Passo 1.

Gera:
    (a) Perfis verticais de qc, qi em diferentes instantes de tempo
    (b) Serie temporal de qc, qi (e qr) integrados na coluna,
        evidenciando a transferencia de massa liquido -> gelo
    (c) Perfil de temperatura, marcando os niveis de 0 graus C e -5 graus C
        (limiar de nucleacao primaria, Cooper 1986)

Para rodar:  python3 simulacao_passo2.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microfisica.coluna_step2 import ColunaFaseGelo
from microfisica.constantes import T0

# ----------------------------------------------------------------------
# 1) Configuracao da coluna: nuvem profunda cruzando o nivel de 0 graus C
# ----------------------------------------------------------------------
coluna = ColunaFaseGelo(nz=80, dz=100.0, T_base=293.0, p_base=95000.0)

k_base = int(1500 / coluna.dz)
k_topo = int(6000 / coluna.dz)
coluna.inserir_nuvem(k_base, k_topo, qc_valor=1.0e-3, Nc_valor=2.0e8)

k_zero_C = int(np.argmin(np.abs(coluna.T - T0)))
k_menos5C = int(np.argmin(np.abs(coluna.T - (T0 - 5.0))))
print(f"Nuvem inserida entre {coluna.z[k_base]:.0f} m e {coluna.z[k_topo]:.0f} m")
print(f"Nivel de 0 graus C:  {coluna.z[k_zero_C]:.0f} m")
print(f"Nivel de -5 graus C (limiar de nucleacao, Cooper 1986): {coluna.z[k_menos5C]:.0f} m")

# ----------------------------------------------------------------------
# 2) Integracao no tempo
# ----------------------------------------------------------------------
TEMPO_TOTAL = 1800.0  # 30 min (suficiente para ver o efeito WBF se desenvolver)
DT = 2.0
historico = coluna.integrar(TEMPO_TOTAL, dt=DT, salvar_a_cada=60.0)

print(f"\nPrecipitacao acumulada na superficie: {coluna.precip_superficie_mm:.4f} mm")

# ----------------------------------------------------------------------
# 3) Grafico (a): perfis verticais de qc e qi em diferentes tempos
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
indices_plot = np.linspace(0, len(historico["t"]) - 1, 6).astype(int)
cores = plt.cm.viridis(np.linspace(0, 1, len(indices_plot)))

for i, idx in enumerate(indices_plot):
    t_min = historico["t"][idx] / 60.0
    axes[0].plot(np.array(historico["qc"][idx]) * 1000, coluna.z,
                 color=cores[i], label=f"t={t_min:.0f} min")
    axes[1].plot(np.array(historico["qi"][idx]) * 1000, coluna.z,
                 color=cores[i], label=f"t={t_min:.0f} min")

for j, ax in enumerate(axes[:2]):
    ax.axhline(coluna.z[k_zero_C], color="gray", ls="--", lw=1, label="0 graus C" if j == 0 else None)
    ax.axhline(coluna.z[k_menos5C], color="tab:red", ls=":", lw=1, label="-5 graus C" if j == 0 else None)
    ax.grid(alpha=0.3)

axes[0].set_xlabel("qc (g/kg)")
axes[0].set_ylabel("Altura (m)")
axes[0].set_title("Agua de nuvem (qc)")
axes[0].legend(fontsize=7)

axes[1].set_xlabel("qi (g/kg)")
axes[1].set_title("Gelo de nuvem (qi)")

axes[2].plot(coluna.T - 273.15, coluna.z, color="black")
axes[2].axhline(coluna.z[k_zero_C], color="gray", ls="--", lw=1)
axes[2].axhline(coluna.z[k_menos5C], color="tab:red", ls=":", lw=1)
axes[2].axvline(0, color="gray", ls="--", lw=0.7)
axes[2].axvline(-5, color="tab:red", ls=":", lw=0.7)
axes[2].set_xlabel("Temperatura ( graus C)")
axes[2].set_title("Perfil de T (final)")
axes[2].grid(alpha=0.3)

plt.suptitle("Passo 2 - Fase gelo e efeito Wegener-Bergeron-Findeisen")
plt.tight_layout()
plt.savefig("./fig_passo2_perfis.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------
# 4) Grafico (b): series temporais integradas na coluna (qc vs qi)
# ----------------------------------------------------------------------
qc_total = [np.sum(np.array(qc) * coluna.rho) * coluna.dz for qc in historico["qc"]]
qi_total = [np.sum(np.array(qi) * coluna.rho) * coluna.dz for qi in historico["qi"]]
qr_total = [np.sum(np.array(qr) * coluna.rho) * coluna.dz for qr in historico["qr"]]
t_min = np.array(historico["t"]) / 60.0

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(t_min, qc_total, label="Agua de nuvem (qc)", color="tab:blue")
ax.plot(t_min, qi_total, label="Gelo de nuvem (qi)", color="tab:cyan")
ax.plot(t_min, qr_total, label="Agua de chuva (qr)", color="tab:orange")
ax.set_xlabel("Tempo (min)")
ax.set_ylabel("Massa total na coluna (kg/m$^2$)")
ax.set_title("Transferencia de massa liquido -> gelo (efeito WBF)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig_passo2_series.png", dpi=130)
plt.close()

print("\nFiguras salvas:")
print(" - ./fig_passo2_perfis.png")
print(" - ./fig_passo2_series.png")

# ----------------------------------------------------------------------
# 5) Dados numericos (perfis completos + series integradas), em ./
# ----------------------------------------------------------------------
np.savez_compressed(
    "./resultados_passo2.npz",
    t_s=np.array(historico["t"]),
    z_m=coluna.z,
    qc=np.array(historico["qc"]),
    qr=np.array(historico["qr"]),
    qi=np.array(historico["qi"]),
    Nc=np.array(historico["Nc"]),
    Nr=np.array(historico["Nr"]),
    Ni=np.array(historico["Ni"]),
    qv=np.array(historico["qv"]),
    T_K=np.array(historico["T"]),
    precip_superficie_mm=coluna.precip_superficie_mm,
)

with open("./series_passo2.csv", "w") as f:
    f.write("tempo_min,qc_total_kgm2,qi_total_kgm2,qr_total_kgm2\n")
    for i in range(len(t_min)):
        f.write(f"{t_min[i]:.4f},{qc_total[i]:.6e},{qi_total[i]:.6e},{qr_total[i]:.6e}\n")

print("\nDados salvos:")
print(" - ./resultados_passo2.npz  (perfis completos: t_s,z_m,qc,qr,qi,Nc,Nr,Ni,qv,T_K)")
print(" - ./series_passo2.csv      (series integradas na coluna)")
