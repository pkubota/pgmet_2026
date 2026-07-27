# -*- coding: utf-8 -*-
"""
MODULO 5  Fluxo de Calor Sensivel [H]
====================================================================
Slide de referencia: "Fluxo de calor sensivel [H]"

    H / (rho*cp) = -(Tr - T) / ra
    ==> H = rho * cp * (T - Tr) / ra

Onde:
    T  = temperatura da superficie/camada de solo (K)  prognostica
    Tr = temperatura do ar no nivel de referencia (K)  forcante
    ra = resistencia aerodinamica (s/m)  modulo 3

Convencao de sinal: H > 0 quando a superficie esta mais quente que
o ar (T > Tr) => fluxo de calor sensivel para cima (perda de energia
pela superficie), igual ao esquema do slide (seta H para cima).
"""

CP_AR = 1004.0  # J kg-1 K-1
RHO_AR = 1.2    # kg m-3


def fluxo_calor_sensivel(T, Tr, ra, rho=RHO_AR, cp=CP_AR):
    """
    H = rho * cp * (T - Tr) / ra
    """
    return rho * cp * (T - Tr) / ra
