# -*- coding: utf-8 -*-
"""
MODULO 8 - Difusao de Umidade no Solo (modelo multicamadas)
====================================================================
Slide de referencia: "Fluxo de calor no solo [G] - c. Modelo de
umidade do solo" (Mahrt e Pan, 1984)

Equacao governante (z crescendo para baixo, positivo = para baixo):

    d(theta)/dt = d/dz [ D(theta) * d(theta)/dz ] + d K(theta)/dz

Forma integrada por camada usada aqui (equivalente ao slide):

    dz_i * d(theta_i)/dt = q_(i-1/2) - q_(i+1/2)

onde q e o fluxo de agua na fase liquida (positivo para baixo):

    q = -D(theta) * d(theta)/dz + K(theta)

D(theta) e K(theta) sao funcoes nao lineares da umidade do solo,
segundo a formulacao classica de Clapp e Hornberger (1978), citada
no slide "O modelo termodinamico de duas camadas de solo":

    K(theta) = Ks * (theta/theta_s)^(2b+3)
    D(theta) = (b*Ks*|psi_s|/theta_s) * (theta/theta_s)^(b+2)

Contorno de superficie: fluxo = precipitacao - evaporacao direta
(E_dir, ver modulos 6/7, convertida para m/s).
Contorno de fundo: drenagem livre, q_fundo = K(theta_fundo).

O esquema numerico e EXPLICITO, com sub-ciclagem automatica interna
para respeitar o criterio de estabilidade (D pode variar em varias
ordens de grandeza entre solo seco e saturado, conforme mencionado
no slide).
"""

import numpy as np


# ---------------------------------------------------------------
# Parametros de solo (Clapp e Hornberger, 1978) - classes texturais
# tipicas usadas em modelos de superficie (ex.: Noah LSM, CLM)
# ---------------------------------------------------------------
PARAMETROS_SOLO = {
    # nome:              (psi_s [m],  theta_s [m3/m3], b [-],  Ks [m/s])
    "areia":             (-0.121, 0.395,  4.05, 1.76e-4),
    "franco_arenoso":    (-0.090, 0.435,  4.38, 1.56e-5),
    "franco":            (-0.218, 0.451,  5.39, 3.47e-6),
    "franco_argiloso":   (-0.259, 0.476,  8.52, 6.30e-6),
    "argila":            (-0.468, 0.482, 11.40, 1.28e-6),
}


def obter_parametros_solo(classe="franco_argiloso"):
    """Retorna (psi_s [m], theta_s [m3/m3], b [-], Ks [m/s])."""
    return PARAMETROS_SOLO[classe]


def difusividade_hidraulica(theta, theta_s, b, Ks, psi_s):
    """D(theta) [m2/s] - Clapp e Hornberger (1978)."""
    theta_r = np.clip(theta / theta_s, 0.02, 1.0)
    return (b * Ks * abs(psi_s) / theta_s) * theta_r ** (b + 2)


def condutividade_hidraulica(theta, theta_s, b, Ks):
    """K(theta) [m/s] - Clapp e Hornberger (1978)."""
    theta_r = np.clip(theta / theta_s, 0.02, 1.0)
    return Ks * theta_r ** (2 * b + 3)


def passo_estavel(dz, D_max, fator_seguranca=0.4):
    """dt_max ~ fator * dz_min^2 / D_max (estabilidade do esquema explicito)."""
    dz_min = np.min(dz)
    if D_max <= 0:
        return 3600.0
    return fator_seguranca * dz_min ** 2 / D_max


def atualizar_umidade_solo(theta, dz, dt, classe_solo="franco_argiloso",
                            evaporacao=0.0, precipitacao=0.0,
                            drenagem_livre=True):
    """
    Avanca o perfil de umidade do solo (multicamadas) por um passo dt,
    usando sub-ciclos explicitos internos.

    Parametros
    ----------
    theta : array (n,)
        Umidade volumetrica de cada camada (m3/m3), do topo para a base.
    dz : array (n,)
        Espessura de cada camada (m).
    dt : float
        Passo de tempo externo (s) - subdividido internamente.
    classe_solo : str
        Classe textural (ver PARAMETROS_SOLO).
    evaporacao : float
        Taxa de evaporacao na superficie (m/s), positivo = saida de agua.
    precipitacao : float
        Taxa de precipitacao (m/s), positivo = entrada de agua.
    drenagem_livre : bool
        Se True, fundo do perfil drena livremente (q = K(theta_fundo)).
        Se False, fundo e impermeavel (q = 0).

    Retorna
    -------
    theta_novo : array (n,)
    """
    psi_s, theta_s, b, Ks = obter_parametros_solo(classe_solo)
    theta = np.array(theta, dtype=float)
    dz = np.array(dz, dtype=float)
    n = len(theta)

    dz_interfaces = 0.5 * (dz[:-1] + dz[1:])

    D_estimado = difusividade_hidraulica(theta, theta_s, b, Ks, psi_s)
    dt_estavel = passo_estavel(dz_interfaces, np.max(D_estimado))
    n_sub = max(1, int(np.ceil(dt / dt_estavel)))
    dt_sub = dt / n_sub

    for _ in range(n_sub):
        D = difusividade_hidraulica(theta, theta_s, b, Ks, psi_s)
        K = condutividade_hidraulica(theta, theta_s, b, Ks)

        D_interf = 0.5 * (D[:-1] + D[1:])
        K_interf = 0.5 * (K[:-1] + K[1:])
        grad = (theta[1:] - theta[:-1]) / dz_interfaces
        q_interf = -D_interf * grad + K_interf  # positivo = para baixo

        q_topo = precipitacao - evaporacao
        q_base = K[-1] if drenagem_livre else 0.0

        q_todos = np.concatenate(([q_topo], q_interf, [q_base]))

        dtheta = (q_todos[:-1] - q_todos[1:]) / dz * dt_sub
        theta = theta + dtheta
        theta = np.clip(theta, 0.02, theta_s * 0.999)

    return theta
