"""
compara_coordenadas.py

Le a saida do programa Fortran advdiff_verticalcoord.f90 (saida_sigma.txt,
saida_z.txt, saida_tempos.txt) e produz uma figura comparando a evolucao do
pulso gaussiano resolvido em coordenada z (Cartesiana) e em coordenada sigma
(Phillips, 1957), ambas mapeadas para a MESMA altura fisica z.

Esta comparacao e o teste pratico de que a transformacao de coordenadas
(csidot, metrica) foi implementada corretamente: como as duas simulacoes
representam a MESMA circulacao fisica (a mesma w(z), obtida uma a partir da
outra pela relacao w = (dz/dsigma)*sigmadot, Secao 2.7.1), os dois perfis,
quando interpolados para o mesmo eixo z, devem coincidir dentro do erro de
truncamento numerico esperado para um esquema de 1a ordem (upwind + Euler).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#12233f"
GOLD = "#b08d2f"
RED = "#a3312a"

# ---------------------------------------------------------------- leitura --
sig = np.loadtxt("saida_sigma.txt")
zc  = np.loadtxt("saida_z.txt")
tsn = np.loadtxt("saida_tempos.txt")

z_sigma = sig[:, 2]                  # altura fisica de cada nivel sigma [m]
A_sigma = {0: sig[:, 3], 1: sig[:, 4], 2: sig[:, 5]}

z_z = zc[:, 1]                       # altura fisica (uniforme) da grade z [m]
A_z = {0: zc[:, 2], 1: zc[:, 3], 2: zc[:, 4]}

t_snap = tsn[:, 1] / 3600.0          # horas

# --------------------------------------------------- grade comum para comparacao --
z_common = np.linspace(0.0, min(z_sigma.max(), z_z.max()), 400)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
labels = [f"t = {t_snap[k]:.1f} h" for k in range(3)]

rms_list = []
for k in range(3):
    ax = axes[k]
    # ordena por z crescente para interpolacao (arquivos estao em ordem de nivel)
    order_s = np.argsort(z_sigma)
    order_z = np.argsort(z_z)

    As_interp = np.interp(z_common, z_sigma[order_s], A_sigma[k][order_s])
    Az_interp = np.interp(z_common, z_z[order_z], A_z[k][order_z])

    rms = np.sqrt(np.mean((As_interp - Az_interp) ** 2))
    rms_list.append(rms)

    ax.plot(A_z[k][order_z], z_z[order_z] / 1000.0, color=NAVY, lw=2.4,
             label="coordenada z (Cartesiana)")
    ax.plot(A_sigma[k][order_s], z_sigma[order_s] / 1000.0, color=GOLD, lw=2.0,
             linestyle="--", marker="o", markersize=3.5,
             label="coordenada σ (Phillips, 1957)")
    ax.set_title(labels[k], fontsize=13, color=NAVY, fontweight="bold")
    ax.set_xlabel("A (escalar transportado)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.text(0.97, 0.03, f"RMS(z − σ) = {rms:.2e}", transform=ax.transAxes,
             ha="right", va="bottom", fontsize=9.5, color=RED,
             bbox=dict(boxstyle="round", fc="white", ec=RED, alpha=0.85))

axes[0].set_ylabel("altura z (km)", fontsize=12)
axes[0].legend(loc="upper right", fontsize=10, framealpha=0.9)
fig.suptitle("Adveccao-difusao vertical: coordenada z vs. coordenada σ — mesma circulação física",
             fontsize=14.5, color=NAVY, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("comparacao_z_sigma.png", dpi=150, bbox_inches="tight")
print("Figura salva em comparacao_z_sigma.png")
print("RMS(z - sigma) por instante:", [f"{r:.3e}" for r in rms_list])

# ------------------------------------------------- figura 2: metrica e w(z)/sigmadot --
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.6))
order_s = np.argsort(z_sigma)
ax2[0].plot(sig[order_s, 1], z_sigma[order_s] / 1000.0, color=NAVY, lw=2)
ax2[0].set_xlabel("σ", fontsize=11)
ax2[0].set_ylabel("altura z (km)", fontsize=11)
ax2[0].set_title("Mapeamento σ ↔ z\n(atmosfera isotérmica, T0=250 K)", fontsize=12, color=NAVY)
ax2[0].grid(alpha=0.25)

ax2[1].plot(A_z[0], z_z / 1000.0, color=GOLD, lw=2, label="t=0 (gaussiana inicial)")
ax2[1].plot(A_z[2], z_z / 1000.0, color=RED, lw=2, label=f"t={t_snap[2]:.1f} h")
ax2[1].set_xlabel("A", fontsize=11)
ax2[1].set_title("Evolução do pulso (coordenada z)", fontsize=12, color=NAVY)
ax2[1].legend(fontsize=10)
ax2[1].grid(alpha=0.25)

fig2.tight_layout()
fig2.savefig("mapeamento_sigma_z.png", dpi=150, bbox_inches="tight")
print("Figura salva em mapeamento_sigma_z.png")
