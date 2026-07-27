# -*- coding: utf-8 -*-
"""
MODULO 2  Fluxo de Calor no Solo [G]  (metodo force-restore)
====================================================================
Slide de referencia: "Fluxo de calor no solo [G]"

    G = (Cs * D / tau_d) * (T - Td)

Onde:
    Cs   = capacidade termica volumetrica do solo (J K-1 m-3)
    D    = espessura da camada de solo (m)
    tau_d = escala de tempo caracteristica (s), tipicamente 1 dia / 2*pi
    T    = temperatura da camada de solo (K)  variavel PROGNOSTICA
    Td   = temperatura do solo profundo (K)  aproximadamente constante
           na escala de um dia ("force-restore": o solo profundo
           "restaura" lentamente a temperatura da camada superficial)

Interpretacao fisica (conforme o slide): quando Rn esta ausente
(a noite), G>0 tende a esfriar a camada superficial em direcao a Td.
"""

import numpy as np

TAU_D_PADRAO = 86400.0 / (2.0 * np.pi)  # 1 dia / (2*pi) em segundos


def fluxo_calor_solo(T, Td, Cs=1.0e6, D=0.1, tau_d=TAU_D_PADRAO):
    """
    Calcula G pelo metodo force-restore.

    Parametros
    ----------
    T     : temperatura da camada de solo (K)
    Td    : temperatura do solo profundo (K)
    Cs    : capacidade termica volumetrica (J K-1 m-3)  [slide: ~1e6]
    D     : espessura da camada (m)
    tau_d : escala de tempo (s)                         [slide: 1 dia/2*pi]
    """
    G = (Cs * D / tau_d) * (T - Td)
    return G
