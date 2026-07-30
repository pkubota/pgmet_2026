"""Figuras do modelo modernizado (fluxos turbulentos + perfis verticais)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modern_column_model import z_a, z_o, hist_simples, hist_mo

NAVY = "#0B2545"
GOLD = "#C9A227"
GRAY = "#6E6E6E"

# ======================================================================
# FIGURA 1: Hovmoller atmosfera (temperatura) - dois fechamentos
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, hist, title in zip(axes, [hist_simples, hist_mo], ["Bulk simples (Cd,Ch,Ce constantes)", "Monin-Obukhov (correção de estabilidade)"]):
    T = hist["Ta"] - 273.15
    cf = ax.contourf(hist["t"], z_a, T.T, levels=25, cmap="RdYlBu_r")
    ax.set_xlabel("tempo (h)")
    ax.set_title(title, fontsize=11)
    for day in [24, 48]:
        ax.axvline(day, color="white", lw=0.8, alpha=0.6)
axes[0].set_ylabel("altura (m)")
cbar = fig.colorbar(cf, ax=axes, shrink=0.85, label="Temperatura do ar (°C)")
fig.suptitle("Evolução do perfil vertical de temperatura na camada limite atmosférica (72 h)", fontsize=13, color=NAVY, y=1.03)
fig.savefig("fig1_hovmoller_atmosfera_temp.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURA 2: Hovmoller atmosfera (umidade especifica)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, hist, title in zip(axes, [hist_simples, hist_mo], ["Bulk simples", "Monin-Obukhov"]):
    Q = hist["qa"] * 1000.0  # g/kg
    cf = ax.contourf(hist["t"], z_a, Q.T, levels=25, cmap="YlGnBu")
    ax.set_xlabel("tempo (h)")
    ax.set_title(title, fontsize=11)
    for day in [24, 48]:
        ax.axvline(day, color="black", lw=0.8, alpha=0.4)
axes[0].set_ylabel("altura (m)")
cbar = fig.colorbar(cf, ax=axes, shrink=0.85, label="Umidade específica (g/kg)")
fig.suptitle("Evolução do perfil vertical de umidade específica na camada limite atmosférica (72 h)", fontsize=13, color=NAVY, y=1.03)
fig.savefig("fig2_hovmoller_atmosfera_umidade.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURA 3: Hovmoller oceano (temperatura)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, hist, title in zip(axes, [hist_simples, hist_mo], ["Bulk simples", "Monin-Obukhov"]):
    T = hist["To"] - 273.15
    cf = ax.contourf(hist["t"], z_o, T.T, levels=25, cmap="RdYlBu_r")
    ax.invert_yaxis()
    ax.set_xlabel("tempo (h)")
    ax.set_title(title, fontsize=11)
    for day in [24, 48]:
        ax.axvline(day, color="white", lw=0.8, alpha=0.6)
axes[0].set_ylabel("profundidade (m)")
cbar = fig.colorbar(cf, ax=axes, shrink=0.85, label="Temperatura da água (°C)")
fig.suptitle("Evolução do perfil vertical de temperatura na camada de mistura oceânica (72 h)", fontsize=13, color=NAVY, y=1.03)
fig.savefig("fig3_hovmoller_oceano_temp.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURA 4: series temporais dos fluxos de superficie e forcante
# ======================================================================
fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

axes[0].plot(hist_simples["t"], hist_simples["solar"], color=GOLD, lw=1.6, label="radiação solar")
axes[0].plot(hist_simples["t"], hist_simples["U"] * 30, color=GRAY, lw=1.3, ls="--", label="vento × 30 (m/s)")
axes[0].set_ylabel("W m$^{-2}$ / (m/s ×30)")
axes[0].legend(fontsize=8, loc="upper right")
axes[0].set_title("Forçante sintética (radiação solar e vento)", fontsize=10.5)

axes[1].plot(hist_simples["t"], hist_simples["H"], color=NAVY, lw=1.6, label="Esq. bulk simples")
axes[1].plot(hist_mo["t"], hist_mo["H"], color=GOLD, lw=1.6, label="Esq. Monin-Obukhov")
axes[1].axhline(0, color="black", lw=0.6)
axes[1].set_ylabel("H (W m$^{-2}$)")
axes[1].set_title("Fluxo de calor sensível oceano→atmosfera", fontsize=10.5)
axes[1].legend(fontsize=8)

axes[2].plot(hist_simples["t"], hist_simples["LE"], color=NAVY, lw=1.6, label="Esq. bulk simples")
axes[2].plot(hist_mo["t"], hist_mo["LE"], color=GOLD, lw=1.6, label="Esq. Monin-Obukhov")
axes[2].set_ylabel("LE (W m$^{-2}$)")
axes[2].set_title("Fluxo de calor latente oceano→atmosfera", fontsize=10.5)
axes[2].legend(fontsize=8)

axes[3].plot(hist_mo["t"], hist_mo["Ch"], color=GOLD, lw=1.6, label="Ch (Monin-Obukhov)")
axes[3].axhline(1.3e-3, color=NAVY, lw=1.6, ls="--", label="Ch (bulk simples, constante)")
axes[3].set_ylabel("Ch")
axes[3].set_xlabel("tempo (h)")
axes[3].set_title("Coeficiente de transferência de calor sensível: efeito da correção de estabilidade", fontsize=10.5)
axes[3].legend(fontsize=8)

for ax in axes:
    ax.grid(alpha=0.3)
    for day in [24, 48]:
        ax.axvline(day, color=GRAY, lw=0.7, ls=":")

fig.tight_layout()
fig.savefig("fig4_series_temporais_fluxos.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("Figuras geradas: fig1_hovmoller_atmosfera_temp.png, fig2_hovmoller_atmosfera_umidade.png,")
print("                 fig3_hovmoller_oceano_temp.png, fig4_series_temporais_fluxos.png")
