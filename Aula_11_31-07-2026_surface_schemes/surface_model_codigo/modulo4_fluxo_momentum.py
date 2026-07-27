# -*- coding: utf-8 -*-
"""
MODULO 4  Fluxo de Momentum [tau]
====================================================================
Slide de referencia: "Fluxo de momentum [tau]"

    tau / rho = -(0 - ur) / ra    (u = 0 na superficie, "no-slip")
    ==> tau = rho * ur / ra

Este modulo apenas reaproveita ra (modulo 3)  mesma "formula geral"
de fluxo turbulento (F = -(f2-f1)/r) aplicada ao momentum.
tau nao entra no balanco de energia do solo, mas e mostrado aqui
por completude didatica, pois integra o "sistema completo" do slide
"O sistema solo-planta-atmosfera".
"""

RHO_AR = 1.2  # kg/m3, densidade do ar ao nivel do mar (aprox.)


def fluxo_momentum(Ur, ra, rho=RHO_AR):
    """
    tau = rho * Ur / ra   (kg m-1 s-2 = Pa)
    """
    return rho * Ur / ra
