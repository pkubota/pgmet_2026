# -*- coding: utf-8 -*-
"""
MODULO 3  Resistencia Aerodinamica [ra]
====================================================================
Slide de referencia: "Resistencia aerodinamica [ra]" (modelo Verma-Rosenberg)

    1/ra = CD * Ur
    CDN  = k^2 / [ ln(zr/z0) ]^2       (condicoes neutras)

Onde:
    CD  = coeficiente de arrasto (adimensional)
    CDN = coeficiente de arrasto em condicoes neutras
    k   = constante de von Karman (~0,4)
    zr  = altura de referencia (m)
    z0  = comprimento de rugosidade (m)
    Ur  = vento no nivel de referencia (m/s)  forcante (modulo 0)

Este e o "resistor" (analogia eletrica, slide "Fluxos turbulentos")
que conecta o nivel de referencia a superficie e que sera reutilizado
nos modulos de momentum, calor sensivel e calor latente.

Simplificacao didatica: assumimos condicoes neutras (CD = CDN), ou
seja, sem correcao de estabilidade  consistente com a hipotese
"C_D = C_DN" listada no slide "Hipoteses*".
"""

import numpy as np

K_VON_KARMAN = 0.4


def coeficiente_arrasto_neutro(zr, z0, k=K_VON_KARMAN):
    """CDN = k^2 / [ln(zr/z0)]^2"""
    return k**2 / (np.log(zr / z0))**2


def resistencia_aerodinamica(Ur, zr=10.0, z0=0.01, k=K_VON_KARMAN, Ur_min=0.5):
    """
    Calcula ra (s/m) supondo condicoes neutras (CD = CDN).

    Parametros
    ----------
    Ur    : vento no nivel de referencia (m/s)
    zr    : altura de referencia (m)
    z0    : comprimento de rugosidade (m)      [slide: ~0,01 m p/ solo nu]
    Ur_min: limite inferior de vento p/ evitar ra -> infinito
    """
    Ur_eff = max(Ur, Ur_min)
    CDN = coeficiente_arrasto_neutro(zr, z0, k)
    ra = 1.0 / (CDN * Ur_eff)
    return ra, CDN
