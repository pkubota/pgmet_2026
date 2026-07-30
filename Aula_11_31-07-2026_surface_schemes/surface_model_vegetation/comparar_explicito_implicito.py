# -*- coding: utf-8 -*-
"""
Compara os metodos EXPLICITO (Euler progressivo) e IMPLICITO (Euler
regressivo linearizado por Taylor, igual ao slide) para a integracao da
temperatura de superficie T no modelo de vegetacao.

Roda 3 simulacoes com as MESMAS forcantes sinteticas de 72h:
  1) IMPLICITO,          dt = 3600 s (1 hora)  -> estavel (usado no modelo)
  2) EXPLICITO,          dt = 3600 s (1 hora)  -> instavel (diverge/oscila)
  3) EXPLICITO,          dt pequeno (< dt_max)  -> estavel, mas com MUITO
     mais passos de tempo para simular o mesmo periodo

Isso ilustra numericamente por que os slides usam o metodo implicito: ele
permite usar o mesmo passo de tempo (1 hora) das rodadas operacionais sem
instabilidade, algo que o metodo explicito so consegue com um passo de
tempo dezenas/centenas de vezes menor (e portanto muito mais caro
computacionalmente).
"""
import numpy as np
import matplotlib.pyplot as plt
from modelo_vegetacao import (rodar_simulacao, gerar_forcante_sintetica,
                               Parametros, passo_explicito, passo_maximo_estavel)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def rodar_explicito_dt_customizado(p, forc_base, dt_pequeno, nhoras=72,
                                    T0=295.0, Td0=294.0, M0=0.0, theta0=0.28):
    """Roda o metodo explicito com um dt menor, interpolando as forcantes
    horarias sinteticas para o passo de tempo desejado (para simular o
    mesmo periodo de 72h com um passo de tempo estavel)."""
    horas_novo = np.arange(0, nhoras + dt_pequeno / 3600.0, dt_pequeno / 3600.0)
    interp = {}
    for chave in ['SWd', 'LWd', 'Tr', 'er', 'Ur', 'P0']:
        interp[chave] = np.interp(horas_novo, forc_base['horas'], forc_base[chave])

    n = len(horas_novo)
    state = np.array([T0, Td0, M0, theta0])
    T_hist = np.zeros(n)
    for i in range(n):
        forc_i = (interp['SWd'][i], interp['LWd'][i], interp['Tr'][i],
                  interp['er'][i], interp['Ur'][i], interp['P0'][i])
        state, _ = passo_explicito(state, forc_i, p, dt_pequeno)
        T_hist[i] = state[0]
    return horas_novo, T_hist


if __name__ == "__main__":
    p = Parametros()

    # ---------------- 1) rodadas com dt = 3600 s (1 hora) -------------------
    forc, hist_imp, _ = rodar_simulacao(p=p, nhoras=72, dt=3600.0,
                                         chuva_dia=2, metodo='implicito')
    _, hist_exp, _ = rodar_simulacao(p=p, nhoras=72, dt=3600.0,
                                      chuva_dia=2, metodo='explicito')

    # ---------------- 2) estimativa do passo maximo estavel -----------------
    estado_ref = np.array([295.0, 294.0, 0.0, 0.28])
    forc_i_meiodia = (forc['SWd'][12], forc['LWd'][12], forc['Tr'][12],
                      forc['er'][12], forc['Ur'][12], forc['P0'][12])
    dt_max = passo_maximo_estavel(estado_ref, forc_i_meiodia, p)
    print(f"dt maximo estavel estimado para o metodo explicito: {dt_max:.1f} s "
          f"({dt_max/60:.2f} min)")
    print(f"dt usado no metodo implicito (estavel):            3600 s (60 min)")
    print(f"razao dt_usado / dt_max_explicito:                 {3600/dt_max:.0f}x")

    # ---------------- 3) rodada do metodo explicito com dt estavel ---------
    dt_estavel = dt_max * 0.5   # margem de seguranca (metade do limite teorico)
    h_exp_ok, T_exp_ok = rodar_explicito_dt_customizado(p, forc, dt_estavel, nhoras=72)

    # ---------------- FIGURA ---------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    h = forc['horas']

    axs[0].plot(h, hist_imp['T'] - 273.15, color="#27ae60", lw=2)
    axs[0].set_ylabel("T (C)")
    axs[0].set_title(f"(A) METODO IMPLICITO (Euler regressivo, Taylor) - dt=3600 s - ESTAVEL")
    axs[0].set_ylim(10, 32)

    axs[1].plot(h, hist_exp['T'] - 273.15, color="#c0392b", lw=1.2)
    axs[1].set_ylabel("T (C)")
    axs[1].set_title(f"(B) METODO EXPLICITO (Euler progressivo) - dt=3600 s - INSTAVEL "
                      f"(dt_max teorico ~ {dt_max:.0f} s)")

    axs[2].plot(h_exp_ok, T_exp_ok - 273.15, color="#2980b9", lw=1.2)
    axs[2].set_ylabel("T (C)")
    axs[2].set_xlabel("Tempo (horas)")
    axs[2].set_title(f"(C) METODO EXPLICITO com dt={dt_estavel:.1f} s (< dt_max) - ESTAVEL, "
                      f"porem com {int(72*3600/dt_estavel)} passos (vs. 72 no caso A)")
    axs[2].set_ylim(10, 32)

    for ax in axs:
        ax.set_xlim(0, 72)
        ax.set_xticks(np.arange(0, 73, 12))

    fig.suptitle("Metodo explicito x implicito na integracao de C dT/dt = Rn-G-H-LE",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("comparacao_explicito_implicito.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Figura salva em comparacao_explicito_implicito.png")
