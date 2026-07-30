# -*- coding: utf-8 -*-
"""
================================================================================
Grupo 3 - MET-579 (Metodos Numericos Aplicados a Modelagem Atmosferica e Oceanica)
Tema: Impacto da resolucao vertical e do esquema de diferencas finitas na
      simulacao de ondas de gravidade internas atmosfericas.

Integrantes: Victor Antunes Ranieri (USP->INPE), Caio Lucas T. F. de Oliveira
             (INPE), Queren Priscila da Silva (UNESP), Paulo Henrique Bazzo
             (UFSC).

Entregavel diferencial: curvas de dispersao sobrepostas (analitica vs.
numerica) para cada configuracao de esquema/resolucao vertical.
================================================================================

FUNDAMENTACAO FISICA
--------------------
Considera-se a atmosfera estratificada, nao-rotante, no regime de Boussinesq
(2D, plano x-z). As equacoes linearizadas para uma perturbacao de onda plana
em torno de um estado basico em repouso, com frequencia de Brunt-Vaisala N
constante, sao:

    du'/dt = -dphi'/dx                                   (quantidade de mov. x)
    dw'/dt = -dphi'/dz + b'                               (quantidade de mov. z)
    db'/dt = -N2 w'                                      (empuxo / termodinamica)
    du'/dx + dw'/dz = 0                                  (continuidade, Boussinesq)

Substituindo o ansatz de onda plana (u',w',b',phi') proporcional a exp[i(kx + mz - omegat)],
obtem-se a relacao de dispersao ANALITICA classica para ondas de gravidade
internas (ver Mesinger & Arakawa, 1976; Kubota, notas de aula MET-576):

                    omega2(k,m) = N2 k2 / (k2 + m2)                          (1)

onde k e o numero de onda horizontal e m o numero de onda vertical.

DISCRETIZACAO VERTICAL E NUMERO DE ONDA MODIFICADO
---------------------------------------------------
O foco do grupo e o impacto da GRADE VERTICAL: mantemos a derivada horizontal
d/dx exata (equivalente a um metodo espectral em x, ou resolucao horizontal
muito superior a vertical) e substituimos d/dz pelo operador de diferencas
finitas efetivamente usado por cada esquema numerico. A tecnica padrao de
"numero de onda modificado" (modified wavenumber analysis) consiste em aplicar
o operador discreto a onda de Fourier e escrever o resultado como
i m* vezes a propria onda, definindo m* (m* = m*(mDeltaz)/Deltaz) como o numero de
onda vertical *efetivo* enxergado pelo esquema. A relacao de dispersao
NUMERICA e obtida substituindo m -> m* em (1):

                    omega2_num(k,m) = N2 k2 / (k2 + m*2)                     (2)

Quatro esquemas sao implementados e suas formulas de m*Deltaz DERIVADAS abaixo
(theta = mDeltaz):

1) Diferencas centradas de 2a ordem, grade nao-alternada (Deltaz):
       f'_j ~= (f_{j+1} - f_{j-1}) / (2Deltaz)
   =>  m*Deltaz = sin(theta)

2) Diferencas centradas de 4a ordem, grade nao-alternada:
       f'_j ~= [-f_{j+2} + 8f_{j+1} - 8f_{j-1} + f_{j-2}] / (12Deltaz)
   =>  m*Deltaz = [8 sin(theta) - sin(2theta)] / 6

3) Esquema compacto (Pade) tridiagonal de 4a ordem (Lele, 1992), alpha=1/4, a=3/2:
       (1/4) f'_{j-1} + f'_j + (1/4) f'_{j+1} = (3/(4Deltaz)) (f_{j+1} - f_{j-1})
   =>  m*Deltaz = 3 sin(theta) / (2 + cos(theta))

4) Diferencas centradas de 2a ordem em GRADE ALTERNADA (staggered, tipo
   Lorenz - a mesma filosofia da grade vertical usada no MPAS/MONAN), com
   derivada avaliada no meio-nivel a partir de pontos separados por
   exatamente Deltaz (nao 2Deltaz):
       f'_{j+1/2} ~= (f_{j+1} - f_j) / Deltaz
   =>  m*Deltaz = 2 sin(theta/2)
   Este operador e EXATO em fase (a derivada cai exatamente no meio-nivel,
   sem vies de fase), sendo por isso menos dispersivo que o esquema (1) para
   o mesmo Deltaz - resultado conhecido de Mesinger & Arakawa (1976) sobre a
   vantagem das grades alternadas na representacao de ondas curtas.

Todas as formulas reduzem-se a m*Deltaz -> theta (isto e, m* -> m, esquema exato)
no limite theta -> 0 (ondas longas / bem resolvidas), e todas divergem do valor
exato a medida que theta -> pi (onda de 2Deltaz, o menor comprimento de onda
representavel na grade), com taxas de convergencia diferentes:
esquemas (1) e (4) sao O(Deltaz2); esquemas (2) e (3) sao O(Deltaz4).

USO
---
Rode este script (`python3 dispersao_ondas_gravidade.py`) para gerar a figura
`dispersao_ondas_gravidade.png` com 4 paineis:
  (a)-(c) curvas de dispersao omega/N vs. numero de onda horizontal adimensional,
          sobrepondo a solucao analitica e as 4 solucoes numericas, para tres
          resolucoes verticais diferentes (onda de 2Deltaz, 4Deltaz e 8Deltaz);
  (d)     curva de convergencia: erro relativo de fase em funcao de mDeltaz,
          para os 4 esquemas, evidenciando a ordem de acuracia de cada um.

Os estudantes podem alterar a lista `resolucoes` (em termos de pontos por
comprimento de onda vertical) e a lista `ESQUEMAS` para explorar outras
configuracoes, conforme pedido no entregavel do grupo.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend nao-interativo (compativel com ambientes headless/JupyterLite)
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) Parametros fisicos de referencia
# ------------------------------------------------------------------
N_BV = 0.012  # frequencia de Brunt-Vaisala [s^-1], valor tipico de troposfera estavel

# ------------------------------------------------------------------
# 2) Numeros de onda modificados (m*Deltaz) para cada esquema, em funcao
#    de theta = m*Deltaz (numero de onda vertical "verdadeiro" x Deltaz)
# ------------------------------------------------------------------
def mstar_dz_2a_ordem(theta):
    """Diferencas centradas, 2a ordem, grade nao-alternada. O(Deltaz2)."""
    return np.sin(theta)


def mstar_dz_4a_ordem(theta):
    """Diferencas centradas, 4a ordem, grade nao-alternada. O(Deltaz4)."""
    return (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0


def mstar_dz_compacto(theta):
    """Esquema compacto (Pade) tridiagonal, 4a ordem (Lele, 1992). O(Deltaz4)."""
    return 3.0 * np.sin(theta) / (2.0 + np.cos(theta))


def mstar_dz_staggered_2a_ordem(theta):
    """Diferencas de 2a ordem em grade alternada tipo Lorenz (estilo MPAS/MONAN). O(Deltaz2)."""
    return 2.0 * np.sin(theta / 2.0)


ESQUEMAS = {
    "Nao-alternada, 2a ordem": dict(func=mstar_dz_2a_ordem, cor="#c0392b", estilo="--"),
    "Nao-alternada, 4a ordem": dict(func=mstar_dz_4a_ordem, cor="#2980b9", estilo="-."),
    "Compacto (Pade), 4a ordem": dict(func=mstar_dz_compacto, cor="#27ae60", estilo=":"),
    "Alternada (Lorenz), 2a ordem": dict(func=mstar_dz_staggered_2a_ordem, cor="#8e44ad", estilo="-"),
}


# ------------------------------------------------------------------
# 3) Relacoes de dispersao analitica e numerica
# ------------------------------------------------------------------
def omega_analitica(k, m, N=N_BV):
    """Eq. (1): relacao de dispersao exata de ondas de gravidade internas."""
    return N * np.abs(k) / np.sqrt(k**2 + m**2)


def omega_numerica(k, m, dz, func_mstar, N=N_BV):
    """Eq. (2): relacao de dispersao numerica, usando m* no lugar de m."""
    theta = m * dz
    mstar = func_mstar(theta) / dz
    return N * np.abs(k) / np.sqrt(k**2 + mstar**2)


# ------------------------------------------------------------------
# 4) Paineis (a)-(c): curvas de dispersao sobrepostas para diferentes
#    resolucoes verticais, expressas em "pontos por comprimento de onda
#    vertical" (Lambdaz/Deltaz = 2pi/theta)
# ------------------------------------------------------------------
resolucoes = [
    dict(theta=np.pi, label="Onda de 2Deltaz  (pior caso resolvido: Nyquist)"),
    dict(theta=np.pi / 2, label="Onda de 4Deltaz"),
    dict(theta=np.pi / 4, label="Onda de 8Deltaz"),
]

dz_ref = 250.0  # m, resolucao vertical de referencia (arbitraria; so fixa a escala)
k_adim = np.linspace(1e-3, 4.0, 400)  # k/m adimensional

fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.38, wspace=0.32)

axes_topo = [fig.add_subplot(gs[0, i]) for i in range(3)]

for ax, res in zip(axes_topo, resolucoes):
    theta0 = res["theta"]
    m = theta0 / dz_ref
    k = k_adim * m  # numero de onda horizontal dimensional

    omega_a = omega_analitica(k, m) / N_BV
    ax.plot(k_adim, omega_a, color="black", lw=2.4, label="Analitica (exata)", zorder=5)

    for nome, cfg in ESQUEMAS.items():
        omega_n = omega_numerica(k, m, dz_ref, cfg["func"]) / N_BV
        ax.plot(k_adim, omega_n, cfg["estilo"], color=cfg["cor"], lw=1.8, label=nome)

    ax.set_title(res["label"], fontsize=11, fontweight="bold")
    ax.set_xlabel(r"$k/m$  (numero de onda horizontal adimensional)")
    ax.set_ylabel(r"$\omega/N$")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

axes_topo[0].legend(loc="lower right", fontsize=8, framealpha=0.9)

# ------------------------------------------------------------------
# 5) Painel (d): curva de convergencia - erro relativo de fase (m*/m - 1)
#    em funcao de theta = mDeltaz, para cada esquema
# ------------------------------------------------------------------
ax_err = fig.add_subplot(gs[1, :])
theta_range = np.linspace(1e-3, np.pi, 400)

for nome, cfg in ESQUEMAS.items():
    mstar_sobre_m = cfg["func"](theta_range) / theta_range
    erro_relativo = np.abs(mstar_sobre_m - 1.0)
    ax_err.semilogy(theta_range / np.pi, erro_relativo, cfg["estilo"], color=cfg["cor"],
                     lw=2.0, label=nome)

ax_err.set_xlabel(r"$m\Delta z\, /\, \pi$   (0 = onda longa/bem resolvida;  1 = onda de 2Deltaz/Nyquist)")
ax_err.set_ylabel(r"Erro relativo em $m^*$ (escala log)")
ax_err.set_title("Convergencia: erro de fase vertical vs. resolucao, por esquema", fontsize=11, fontweight="bold")
ax_err.grid(alpha=0.3, which="both")
ax_err.legend(loc="upper left", fontsize=9)

fig.suptitle(
    "Impacto da resolucao vertical e do esquema de diferencas finitas\n"
    "na simulacao de ondas de gravidade internas atmosfericas -- Grupo 3, MET-579",
    fontsize=13, fontweight="bold", y=0.995,
)

fig.savefig("dispersao_ondas_gravidade.png", dpi=160, bbox_inches="tight")
print("Figura salva em dispersao_ondas_gravidade.png")

# ------------------------------------------------------------------
# 6) Tabela-resumo: erro relativo no pior caso (onda de 2Deltaz, theta=pi)
#    e no caso de 4Deltaz (theta=pi/2), util para discussao quantitativa
# ------------------------------------------------------------------
print("\nResumo do erro relativo em m* (theta = m*Deltaz):")
print(f"{'Esquema':35s}{'2Dz (th=pi)':>16s}{'4Dz (th=pi/2)':>16s}{'8Dz (th=pi/4)':>16s}")
for nome, cfg in ESQUEMAS.items():
    linha = f"{nome:35s}"
    for theta0 in (np.pi, np.pi / 2, np.pi / 4):
        val = abs(cfg["func"](theta0) / theta0 - 1.0)
        linha += f"{val*100:15.2f}%"
    print(linha)
