# -*- coding: utf-8 -*-
"""Gera as figuras didaticas do modelo com vegetacao (ciclo diurno de 72h).

Versao para uso interativo no JupyterLab / Jupyter Notebook:
  - as figuras sao exibidas inline (nao apenas salvas em arquivo);
  - os arquivos .png sao salvos na pasta atual (mesma pasta do notebook).
"""
import numpy as np
import matplotlib.pyplot as plt
from modelo_vegetacao import rodar_simulacao, Parametros

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def sombrear_noites(ax, horas):
    for d in range(int(horas.max() // 24) + 2):
        ax.axvspan(24 * d + 18, 24 * d + 30, color="0.88", zorder=0, lw=0)


def eixo_padrao(ax, h):
    sombrear_noites(ax, h)
    ax.set_xlim(0, h.max())
    ax.set_xticks(np.arange(0, h.max() + 1, 12))


def figura_forcantes(forc, fname):
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    h = forc['horas']

    axs[0].plot(h, forc['SWd'], color="#e67e22", lw=1.8, label="SWd (onda curta)")
    axs[0].plot(h, forc['LWd'], color="#c0392b", lw=1.4, ls="--", label="LWd (onda longa)")
    axs[0].set_ylabel("W/m2")
    axs[0].set_title("Radiacao incidente (SWd, LWd)")
    axs[0].legend(fontsize=9)

    axs[1].plot(h, forc['Tr'] - 273.15, color="#c0392b", lw=1.8)
    axs[1].set_ylabel("Tr ( grausC)")
    axs[1].set_title("Temperatura do ar no nivel de referencia")

    axs[2].plot(h, forc['RH'] * 100, color="#2980b9", lw=1.8)
    axs[2].set_ylabel("UR (%)")
    axs[2].set_title("Umidade relativa do ar")

    axs[3].bar(h, forc['P0'] * 3600, width=1.0, color="#3498db", label="P0 (chuva no topo do dossel)")
    axs[3].set_ylabel("Chuva (mm/h)")
    ax2 = axs[3].twinx()
    ax2.plot(h, forc['Ur'], color="#7f8c8d", lw=1.5, label="Ur (vento)")
    ax2.set_ylabel("Vento (m/s)")
    axs[3].set_title("Precipitacao sintetica (P0) e vento (Ur)")
    axs[3].set_xlabel("Tempo (horas)")

    for ax in axs:
        eixo_padrao(ax, h)

    fig.suptitle("Forcantes sinteticas do ciclo diurno (72 h / 3 dias)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def figura_balanco_hidrico(forc, hist, p, fname):
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    h = forc['horas']

    S = p.S_por_LAI * p.LAI
    axs[0].plot(h, hist['M'], color="#2980b9", lw=2, label=f"M (agua interceptada, S={S:.2f} mm)")
    ax2 = axs[0].twinx()
    ax2.plot(h, hist['W'], color="#16a085", lw=1.6, ls="--", label="W = M/S")
    ax2.set_ylabel("W (fracao molhada)")
    ax2.set_ylim(-0.05, 1.05)
    axs[0].set_ylabel("M (mm)")
    axs[0].set_title("Interceptacao: agua armazenada na copa (M) e fracao molhada (W=M/S)")
    l1, lb1 = axs[0].get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    axs[0].legend(l1 + l2, lb1 + lb2, fontsize=9, loc="upper right")

    axs[1].plot(h, hist['theta'] * 100, color="#8e44ad", lw=2, label="theta (umidade do solo)")
    axs[1].axhline(p.theta_sat * 100, color="0.5", lw=1, ls=":", label="theta saturacao")
    axs[1].axhline(p.theta_wilt * 100, color="0.5", lw=1, ls="-.", label="theta ponto de murcha")
    axs[1].set_ylabel("theta (% vol.)")
    axs[1].set_title("Umidade volumetrica do solo (reservatorio de profundidade D)")
    axs[1].legend(fontsize=9, ncol=3)

    axs[2].plot(h, hist['P1'] * 3600, color="#3498db", lw=1.6, label="P1 (throughfall)")
    axs[2].plot(h, hist['Ru'] * 3600, color="#d35400", lw=1.6, label="Ru (drenagem profunda)")
    axs[2].plot(h, hist['Rs'] * 3600, color="#c0392b", lw=1.6, label="Rs (escoamento superficial)")
    axs[2].set_ylabel("mm/h")
    axs[2].set_title("Termos do balanco hidrico do solo: P1, Rs, Ru")
    axs[2].legend(fontsize=9, ncol=3)
    axs[2].set_xlabel("Tempo (horas)")

    for ax in axs:
        eixo_padrao(ax, h)

    fig.suptitle("Balanco hidrico com vegetacao: interceptacao e umidade do solo", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def figura_energia(forc, hist, fname):
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    h = forc['horas']

    axs[0].plot(h, hist['T'] - 273.15, color="#c0392b", lw=2, label="T (superficie: dossel+solo)")
    axs[0].plot(h, hist['Td'] - 273.15, color="#8e44ad", lw=2, label="Td (solo profundo)")
    axs[0].plot(h, forc['Tr'] - 273.15, color="#7f8c8d", lw=1.2, ls="--", label="Tr (ar)")
    axs[0].set_ylabel("Temperatura ( grausC)")
    axs[0].set_title("Temperaturas prognosticas (solucao implicita backward)")
    axs[0].legend(fontsize=9)

    ET = hist['LEi'] + hist['LEc'] + hist['LEs']
    axs[1].plot(h, hist['Rn'], color="black", lw=2, label="Rn")
    axs[1].plot(h, hist['H'], color="#c0392b", lw=1.6, label="H")
    axs[1].plot(h, ET, color="#2980b9", lw=1.6, label="L(Ei+Ec+Es)")
    axs[1].plot(h, hist['G'], color="#8e44ad", lw=1.6, label="G")
    axs[1].axhline(0, color="0.4", lw=0.8)
    axs[1].set_ylabel("W/m2")
    axs[1].set_title("Balanco de energia da superficie: C dT/dt = Rn - G - H - L(Ei+Ec+Es)")
    axs[1].legend(fontsize=9, ncol=2)

    axs[2].plot(h, hist['LEi'], color="#2980b9", lw=1.6, label="LEi (evap. agua interceptada)")
    axs[2].plot(h, hist['LEc'], color="#27ae60", lw=1.8, label="LEc (transpiracao)")
    axs[2].plot(h, hist['LEs'], color="#d35400", lw=1.6, label="LEs (evaporacao do solo)")
    axs[2].set_ylabel("W/m2")
    axs[2].set_title("Particao do calor latente: interceptacao x transpiracao x solo")
    axs[2].legend(fontsize=9, ncol=3)
    axs[2].set_xlabel("Tempo (horas)")

    for ax in axs:
        eixo_padrao(ax, h)

    fig.suptitle("Balanco de energia da superficie vegetada", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def figura_resistencias(forc, hist, fname):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    h = forc['horas']

    axs[0].plot(h, hist['ra'], color="#16a085", lw=1.8, label="ra (aerodinamica)")
    axs[0].plot(h, hist['rc'], color="#e67e22", lw=1.8, label="rc = rs/LAI (copa)")
    axs[0].plot(h, hist['rs'], color="#c0392b", lw=1.2, ls="--", label="rs (estomatica, por folha)")
    axs[0].plot(h, hist['rsoil'], color="#8e44ad", lw=1.4, ls=":", label="rsoil (solo)")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Resistencia (s/m, escala log)")
    axs[0].set_title("Resistencias: aerodinamica (ra), de copa (rc) e do solo (rsoil)")
    axs[0].legend(fontsize=9, ncol=2)

    axs[1].plot(h, hist['tau'], color="#2c3e50", lw=1.8)
    axs[1].set_ylabel("tau (N/m2)")
    axs[1].set_title("Fluxo de momentum (tau = rho*Ur/ra)")
    axs[1].set_xlabel("Tempo (horas)")

    for ax in axs:
        eixo_padrao(ax, h)

    fig.suptitle("Resistencias e fluxo de momentum ao longo do ciclo diurno", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    forc, hist, p = rodar_simulacao(nhoras=72, dt=3600.0, chuva_dia=2)

    figura_forcantes(forc, "veg_fig1_forcantes.png")
    figura_balanco_hidrico(forc, hist, p, "veg_fig2_balanco_hidrico.png")
    figura_energia(forc, hist, "veg_fig3_balanco_energia.png")
    figura_resistencias(forc, hist, "veg_fig4_resistencias.png")

    # checagem do balanco de energia (deve corresponder ao termo de
    # armazenamento de calor C*dT/dt, nao a zero)
    Rn = hist['Rn']; H = hist['H']; LE = hist['LEi'] + hist['LEc'] + hist['LEs']; G = hist['G']
    dS_diag = Rn - H - LE - G
    dT = np.diff(hist['T'], prepend=hist['T'][0])
    dS_check = p.C / forc['dt'] * dT
    erro = np.mean(np.abs(dS_diag - dS_check))
    print("Erro de fechamento do balanco de energia (deve ser ~0, W/m2):", erro)
    print("Figuras geradas com sucesso.")
