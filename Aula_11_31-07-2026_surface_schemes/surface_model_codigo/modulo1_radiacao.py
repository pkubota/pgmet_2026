# -*- coding: utf-8 -*-
"""
MODULO 1  Saldo de Radiacao [Rn]
====================================================================
Slide de referencia: "Saldo de radiacao [Rn]"

    Rn = SWd - SWu + LWd - LWu
    SWu = alpha * SWd                (onda curta refletida)
    LWu = epsilon * sigma * T^4      (onda longa emitida pela superficie)

    ==>  Rn = (1 - alpha) * SWd + LWd - epsilon * sigma * T^4

Onde:
    alpha       = albedo da superficie (adimensional, 0-1)
    epsilon     = emissividade da superficie (adimensional, 0-1)
    sigma       = constante de Stefan-Boltzmann (5,67e-8 W m-2 K-4)
    T           = temperatura da superficie (K)  variavel PROGNOSTICA
    SWd, LWd    = forcantes atmosfericas (modulo 0)
"""

SIGMA = 5.67e-8  # W m-2 K-4


def saldo_radiacao(SWd, LWd, T, albedo=0.30, emissividade=0.97, sigma=SIGMA):
    """
    Calcula o saldo de radiacao Rn e os termos refletido/emitido.

    Retorna
    -------
    Rn, SWu, LWu : float
    """
    SWu = albedo * SWd
    LWu = emissividade * sigma * T**4
    Rn = SWd - SWu + LWd - LWu
    return Rn, SWu, LWu
