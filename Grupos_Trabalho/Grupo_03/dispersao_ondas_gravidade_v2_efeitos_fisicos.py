# -*- coding: utf-8 -*-
"""
================================================================================
Grupo 3 - MET-579 -- PARTE 2: efeitos fisicos adicionais para maior realismo
Tema: Impacto da resolucao vertical e do esquema de diferencas finitas na
      simulacao de ondas de gravidade internas atmosfericas.

Este script ESTENDE o modelo da Parte 1 (dispersao_ondas_gravidade.py), que
tratava apenas o caso mais idealizado (Boussinesq, sem rotacao, sem vento de
fundo). Aqui incorporamos tres efeitos fisicos discutidos no Capitulo 7 do
Holton (An Introduction to Dynamic Meteorology) para tornar o modelo mais
realista, mantendo a mesma tecnica de numero de onda vertical modificado (m*)
ja usada na Parte 1 para representar cada esquema/resolucao vertical.

Os tres efeitos adicionados sao:

  (A) ROTACAO DA TERRA -> ondas de inercia-gravidade
  (B) VENTO ZONAL MEDIO DE FUNDO -> deslocamento Doppler e nivel critico
  (C) VELOCIDADE DE GRUPO VERTICAL -> transporte de energia (analitico vs.
      numerico), grandeza tao importante quanto a frequencia para a validade
      fisica da simulacao, pois e ela (e nao a velocidade de fase) que
      transporta energia.

--------------------------------------------------------------------------------
(A) ROTACAO: ONDAS DE INERCIA-GRAVIDADE
--------------------------------------------------------------------------------
Partindo do mesmo conjunto de equacoes de Boussinesq 2D usado na Parte 1, mas
agora incluindo o termo de Coriolis (com uma componente meridional v'
acoplada por f), o sistema linearizado e:

    du'/dt - f v' = -dphi'/dx
    dv'/dt + f u'  = 0
    dw'/dt         = -dphi'/dz + b'
    db'/dt         = -N2 w'
    du'/dx + dw'/dz = 0

Resolvendo o sistema algebrico para o ansatz de onda plana exp[i(kx+mz-omegat)]
(eliminando u', v', w', b' e phi' passo a passo) chega-se a relacao de dispersao
NAO-HIDROSTATICA completa para ondas de inercia-gravidade:

                 omega2(k,m) = (N2 k2 + f2 m2) / (k2 + m2)                    (A1)

Esta expressao se reduz a duas situacoes-limite conhecidas:
  - f -> 0:            recupera a relacao de ondas de gravidade puras (Parte 1);
  - k2 << m2 (hidrostatico): omega2 -> f2 + N2k2/m2, o limite hidrostatico classico.
Ambos os limites servem como teste de consistencia da formula (A1).

Fisicamente, (A1) mostra que a frequencia de ondas de inercia-gravidade fica
sempre confinada a faixa f <= |omega| <= N (com N > f nas condicoes troposfericas
usuais) -- ou seja, a rotacao impoe um "piso" de frequencia que simplesmente
nao existe no modelo original sem rotacao. Para os periodos tipicos de
mesoescala (minutos a poucas horas), esse piso costuma ser irrelevante, mas
para ondas de periodo mais longo (dezenas de horas) a rotacao passa a
dominar a dinamica.

Versao NUMERICA: como antes, discretizamos apenas d/dz (o foco do grupo e a
resolucao vertical), substituindo m -> m* nas DUAS ocorrencias de m em (A1):

                 omega2_num = (N2 k2 + f2 m*2) / (k2 + m*2)                   (A2)


--------------------------------------------------------------------------------
(B) VENTO ZONAL MEDIO DE FUNDO: DESLOCAMENTO DOPPLER E NIVEL CRITICO
--------------------------------------------------------------------------------
Um vento basico U (constante, na direcao x) simplesmente desloca a frequencia
observada por um referencial fixo em relacao a frequencia intrinseca (a que
seria vista por um observador movendo-se com o escoamento): substituindo
d/dt -> d/dt + Ud/dx nas equacoes da Parte 1, a relacao de dispersao vira

                 omega = U k +/- omega(k,m),      omega(k,m) = N|k| / sqrt(k2+m2)       (B1)

Esse deslocamento Doppler e importante para casos realistas como ondas de
montanha (lee waves): uma onda estacionaria em relacao ao terreno (omega=0) so
existe enquanto a frequencia intrinseca |omega| = |Uk| permanecer abaixo de N;
quando |Uk| > N a onda deixa de se propagar verticalmente e passa a decair
com a altura (fica "presa"/evanescente) -- este e o chamado NIVEL CRITICO.
A versao numerica troca novamente m -> m* dentro de omega.


--------------------------------------------------------------------------------
(C) VELOCIDADE DE GRUPO VERTICAL: TRANSPORTE DE ENERGIA
--------------------------------------------------------------------------------
A energia de uma onda de gravidade se propaga na velocidade de GRUPO, nao na
de fase. Derivando omega(k,m) = N k/sqrt(k2+m2) (Parte 1) em relacao a m:

            c_gz(k,m) = domega/dm = - N k m / (k2 + m2)^(3/2)                (C1)

Um esquema numerico que representa mal m (isto e, com m* != m) tambem erra a
velocidade de grupo vertical -- e esse erro pode ser proporcionalmente maior
do que o erro na frequencia, ja que a velocidade de grupo depende da
DERIVADA da relacao de dispersao, sendo mais sensivel a erros de fase locais.
Isso e relevante porque erros na velocidade de grupo significam transporte de
energia incorreto (por exemplo, energia de ondas de montanha chegando a
estratosfera mais cedo/mais tarde ou com amplitude errada do que a fisica
real preve). Aqui calculamos c_gz numerica por diferenciacao numerica direta
da propria relacao de dispersao discreta (2), o que evita ter que derivar
analiticamente cada esquema.

USO
---
Rode `python3 dispersao_ondas_gravidade_v2_efeitos_fisicos.py` para gerar
`efeitos_fisicos_ondas_gravidade.png`, com os tres novos efeitos ilustrados
lado a lado com o mesmo conjunto de 4 esquemas numericos da Parte 1. Os
estudantes podem alterar `LAT_GRAUS` (latitude, define f) e `U_VENTO` (vento
basico) para explorar outros regimes.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0) Parametros fisicos
# ------------------------------------------------------------------
N_BV = 0.012          # frequencia de Brunt-Vaisala [s^-1] (troposfera estavel)
OMEGA_TERRA = 7.292e-5  # velocidade angular da Terra [s^-1]
LAT_GRAUS = 45.0        # latitude de referencia (estudantes podem variar)
F_CORIOLIS = 2.0 * OMEGA_TERRA * np.sin(np.deg2rad(LAT_GRAUS))
U_VENTO = 15.0          # vento zonal basico de referencia [m/s] (ondas de montanha)
dz_ref = 250.0          # resolucao vertical de referencia [m]

print(f"f (Coriolis) em {LAT_GRAUS:.0f} graus = {F_CORIOLIS:.2e} s^-1  "
      f"(periodo inercial = {2*np.pi/F_CORIOLIS/3600:.1f} h)")
print(f"N/f = {N_BV/F_CORIOLIS:.1f}  (N deve ser >> f para a aproximacao ser valida)")

# ------------------------------------------------------------------
# 1) Numeros de onda modificados (m*Delta z), reaproveitados da Parte 1
# ------------------------------------------------------------------
def mstar_dz_2a_ordem(theta):
    return np.sin(theta)


def mstar_dz_4a_ordem(theta):
    return (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0


def mstar_dz_compacto(theta):
    return 3.0 * np.sin(theta) / (2.0 + np.cos(theta))


def mstar_dz_staggered_2a_ordem(theta):
    return 2.0 * np.sin(theta / 2.0)


ESQUEMAS = {
    "Nao-alternada, 2a ordem": dict(func=mstar_dz_2a_ordem, cor="#c0392b", estilo="--"),
    "Nao-alternada, 4a ordem": dict(func=mstar_dz_4a_ordem, cor="#2980b9", estilo="-."),
    "Compacto (Pade), 4a ordem": dict(func=mstar_dz_compacto, cor="#27ae60", estilo=":"),
    "Alternada (Lorenz), 2a ordem": dict(func=mstar_dz_staggered_2a_ordem, cor="#8e44ad", estilo="-"),
}


def mstar(m, dz, func_mstar):
    """Numero de onda vertical efetivo m* enxergado pelo esquema discreto."""
    return func_mstar(m * dz) / dz


# ------------------------------------------------------------------
# (A) Ondas de inercia-gravidade: analitica e numerica -- Eqs. (A1)-(A2)
# ------------------------------------------------------------------
def omega_rotacao_analitica(k, m, f=F_CORIOLIS, N=N_BV):
    return np.sqrt((N**2 * k**2 + f**2 * m**2) / (k**2 + m**2))


def omega_rotacao_numerica(k, m, dz, func_mstar, f=F_CORIOLIS, N=N_BV):
    ms = mstar(m, dz, func_mstar)
    return np.sqrt((N**2 * k**2 + f**2 * ms**2) / (k**2 + ms**2))


# ------------------------------------------------------------------
# (B) Deslocamento Doppler por vento medio de fundo -- Eq. (B1)
# ------------------------------------------------------------------
def omega_intrinseca_analitica(k, m, N=N_BV):
    return N * np.abs(k) / np.sqrt(k**2 + m**2)


def omega_observada_analitica(k, m, U=U_VENTO, N=N_BV):
    return U * k + omega_intrinseca_analitica(k, m, N)


def omega_observada_numerica(k, m, dz, func_mstar, U=U_VENTO, N=N_BV):
    ms = mstar(m, dz, func_mstar)
    return U * k + N * np.abs(k) / np.sqrt(k**2 + ms**2)


# ------------------------------------------------------------------
# (C) Velocidade de grupo vertical -- analitica (Eq. C1) e numerica
#     (diferenciacao numerica direta da relacao de dispersao discreta)
# ------------------------------------------------------------------
def cgz_analitica(k, m, N=N_BV):
    return -N * k * m / (k**2 + m**2) ** 1.5


def cgz_numerica(k, m, dz, func_mstar, N=N_BV, dm_frac=1e-4):
    """domega_num/dm por diferenca central, aplicada diretamente sobre a
    relacao de dispersao discreta (nao sobre a formula continua)."""
    dm = dm_frac * max(abs(m), 1.0 / dz)
    om_mais = N * np.abs(k) / np.sqrt(k**2 + mstar(m + dm, dz, func_mstar) ** 2)
    om_menos = N * np.abs(k) / np.sqrt(k**2 + mstar(m - dm, dz, func_mstar) ** 2)
    return (om_mais - om_menos) / (2 * dm)


# ==================================================================
# FIGURA: 3 paineis lado a lado, um para cada efeito fisico
# ==================================================================
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5.2))

# ---------------- Painel A: ondas de inercia-gravidade -----------
theta0 = np.pi / 2  # onda de 4 Delta z, resolucao intermediaria
m = theta0 / dz_ref
k_dim = np.linspace(1e-3, 4.0, 400) * m

omega_a = omega_rotacao_analitica(k_dim, m) / N_BV
axA.plot(k_dim / m, omega_a, color="black", lw=2.4, label="Analitica", zorder=5)
for nome, cfg in ESQUEMAS.items():
    om_n = omega_rotacao_numerica(k_dim, m, dz_ref, cfg["func"]) / N_BV
    axA.plot(k_dim / m, om_n, cfg["estilo"], color=cfg["cor"], lw=1.7, label=nome)
axA.axhline(F_CORIOLIS / N_BV, color="gray", lw=1, ls=":", label="Piso omega=f")
axA.set_xlabel(r"$k/m$")
axA.set_ylabel(r"$\omega/N$")
axA.set_title("(A) Ondas de inercia-gravidade\n(onda de 4Deltaz, f em 45 graus N)", fontsize=10, fontweight="bold")
axA.legend(fontsize=6.5, loc="lower right")
axA.grid(alpha=0.3)

# ---------------- Painel B: deslocamento Doppler ------------------
# Escala de k baseada no numero de onda critico k_c = N/U (onde a frequencia
# intrinseca de uma onda estacionaria se aproxima de N); m mantido no valor
# de resolucao vertical do painel A (onda de 4 Delta z).
k_c = N_BV / U_VENTO
k_signed = np.linspace(-4.0, 4.0, 400) * k_c
om_obs_a = omega_observada_analitica(k_signed, m) / N_BV
axB.plot(k_signed / k_c, om_obs_a, color="black", lw=2.4, label="Analitica", zorder=5)
for nome, cfg in ESQUEMAS.items():
    om_obs_n = omega_observada_numerica(k_signed, m, dz_ref, cfg["func"]) / N_BV
    axB.plot(k_signed / k_c, om_obs_n, cfg["estilo"], color=cfg["cor"], lw=1.7, label=nome)
axB.set_xlabel(r"$k/k_c$,  $k_c \equiv N/U$  (negativo = propagacao para oeste)")
axB.set_ylabel(r"$\omega_{observada}/N$")
axB.set_title(f"(B) Deslocamento Doppler\n(vento de fundo U={U_VENTO:.0f} m/s)", fontsize=10, fontweight="bold")
axB.legend(fontsize=6.5, loc="upper left")
axB.grid(alpha=0.3)

# ---------------- Painel C: velocidade de grupo vertical ----------
theta_range = np.linspace(0.05, np.pi * 0.98, 200)
k_fixo = m  # onda isotropica k=m para ilustrar o efeito claramente

cg_a = [cgz_analitica(k_fixo, th / dz_ref) for th in theta_range]
axC.plot(theta_range / np.pi, cg_a, color="black", lw=2.4, label="Analitica", zorder=5)
for nome, cfg in ESQUEMAS.items():
    cg_n = [cgz_numerica(k_fixo, th / dz_ref, dz_ref, cfg["func"]) for th in theta_range]
    axC.plot(theta_range / np.pi, cg_n, cfg["estilo"], color=cfg["cor"], lw=1.7, label=nome)
axC.set_xlabel(r"$m\Delta z / \pi$")
axC.set_ylabel(r"$c_{gz}$  [m/s]")
axC.set_title("(C) Velocidade de grupo vertical\n(transporte de energia, k=m)", fontsize=10, fontweight="bold")
axC.legend(fontsize=6.5, loc="lower left")
axC.grid(alpha=0.3)

fig.suptitle(
    "Efeitos fisicos adicionais para maior realismo -- Grupo 3, MET-579\n"
    "(rotacao, vento de fundo e velocidade de grupo; base fisica: Holton, Cap. 7)",
    fontsize=12, fontweight="bold", y=1.04,
)
fig.tight_layout()
fig.savefig("efeitos_fisicos_ondas_gravidade.png", dpi=160, bbox_inches="tight")
print("\nFigura salva em efeitos_fisicos_ondas_gravidade.png")
