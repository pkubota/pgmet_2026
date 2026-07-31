# -*- coding: utf-8 -*-
"""
distribuicoes.py
=================

Funcoes relacionadas a distribuicao de tamanhos (Particle Size
Distribution, PSD) assumida para cada categoria de hidrometeoro.

TEORIA
------
Cada categoria de hidrometeoro (goticulas de nuvem, chuva, gelo, neve,
...) e representada, num esquema "bulk" de DOIS MOMENTOS, por apenas
duas variaveis prognosticas:

    q  = razao de mistura de massa   [kg de agua / kg de ar seco]
    N  = concentracao numerica total [numero de particulas / kg de ar seco]

So com (q, N) nao sabemos a forma exata do espectro de tamanhos -- por
isso ASSUMIMOS que ele segue uma distribuicao gama:

    N(D) = N0 * D^mu * exp(-lambda * D)        [m^-4]

onde:
    D      = diametro da particula (m)
    N0     = parametro de intercepto (m^-(mu+1))
    mu     = parametro de forma (adimensional, fixado por categoria)
    lambda = parametro de inclinacao (m^-1)

Dado mu (fixo) e as duas variaveis prognosticas (q, N), podemos
DIAGNOSTICAR (N0, lambda) em cada passo de tempo. E exatamente isso
que o MG2008 faz (e o Thompson et al. tambem, com pequenas variacoes).

DEDUCAO DE lambda A PARTIR DE (q, N)
-------------------------------------
Concentracao numerica total (0-esimo momento):

    N = integral_0^inf N(D) dD = N0 * Gamma(mu+1) / lambda^(mu+1)   (i)

Razao de mistura de massa, assumindo particulas esfericas de densidade
rho_x (3o momento, escalado por rho_x*pi/6/rho_ar):

    q = (pi*rho_x)/(6*rho_ar) * integral_0^inf D^3 N(D) dD
      = (pi*rho_x)/(6*rho_ar) * N0 * Gamma(mu+4) / lambda^(mu+4)     (ii)

Substituindo N0 de (i) em (ii) e isolando lambda:

    lambda^3 = [pi * rho_x * Gamma(mu+4) * N] / [6 * rho_ar * Gamma(mu+1) * q]

    lambda = { [pi * rho_x * Gamma(mu+4) * N] /
               [6 * rho_ar * Gamma(mu+1) * q] } ** (1/3)

Esta e a relacao central usada por TODOS os processos de microfisica
que dependem do tamanho das particulas (autoconversao, acrecao,
sedimentacao, etc.), pois ela conecta as variaveis prognosticas (q, N)
a forma assumida do espectro.
"""

import numpy as np
from .constantes import gamma_func, QMIN, NMIN


def lambda_gama(q, N, rho_ar, rho_x, mu):
    """
    Calcula o parametro de inclinacao (lambda) da distribuicao gama.

    Parametros
    ----------
    q : float
        Razao de mistura de massa da categoria (kg/kg).
    N : float
        Concentracao numerica da categoria (kg^-1, i.e. # por kg de ar).
    rho_ar : float
        Densidade do ar (kg/m^3).
    rho_x : float
        Densidade "bulk" assumida para a especie (kg/m^3).
    mu : float
        Parametro de forma da distribuicao gama (fixo, adimensional).

    Retorna
    -------
    lam : float
        Parametro de inclinacao lambda (m^-1). Se q ou N forem
        despreziveis, retorna um valor grande (particulas "inexistentes"
        -> distribuicao degenerada, sem efeito nos processos).
    """
    if q <= QMIN or N <= NMIN:
        return 1.0e10  # particulas efetivamente ausentes

    numer = np.pi * rho_x * gamma_func(mu + 4.0) * N
    denom = 6.0 * rho_ar * gamma_func(mu + 1.0) * q
    lam = (numer / denom) ** (1.0 / 3.0)
    return lam


def N0_gama(N, lam, mu):
    """Parametro de intercepto N0 = N * lambda^(mu+1) / Gamma(mu+1)."""
    return N * lam ** (mu + 1.0) / gamma_func(mu + 1.0)


def diametro_medio_massa(lam, mu):
    """
    Diametro medio ponderado pela massa (Dm), util como diagnostico e
    para plots comparativos com os slides/artigos.

        Dm = (mu + 4) / lambda
    """
    return (mu + 4.0) / lam


def diametro_medio_numero(lam, mu):
    """Diametro medio ponderado pelo numero: Dn = (mu+1)/lambda."""
    return (mu + 1.0) / lam
