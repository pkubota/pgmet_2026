# -*- coding: utf-8 -*-
"""
MODULO 7  Balanco de Agua do Solo (bucket model)
====================================================================
Slide de referencia: "Balanco de agua do solo"

    D * d(theta)/dt = P - E - (Rs + Ru) = P - E - R

Neste modelo diurno simplificado (sem precipitacao prescrita), o
unico sumidouro de agua e a evaporacao E, obtida a partir de LE
(modulo 6):

    E [m/s] = LE / (L * rho_agua)

O runoff R so e ativado quando o "balde" transborda (theta > theta_s),
conforme a regra do "bucket model" do slide:
    P >= E, theta_s  -> R = P - E  (transborda)
    P <  E            -> R = 0
"""

import numpy as np

L_VAPORIZACAO = 2.5e6  # J kg-1
RHO_AGUA = 1000.0      # kg m-3


def balanco_hidrico_solo(theta, LE, dt, D_solo=0.5, P=0.0,
                          L=L_VAPORIZACAO, rho_agua=RHO_AGUA,
                          theta_min=0.02, theta_max=0.50):
    """
    Atualiza o conteudo volumetrico de agua do solo (theta) por um
    passo de tempo dt (s), dado o fluxo de calor latente LE (W/m2).

    Parametros
    ----------
    theta   : umidade volumetrica atual (m3/m3)
    LE      : fluxo de calor latente (W/m2)  vindo do modulo 6
    dt      : passo de tempo (s)
    D_solo  : espessura da camada de solo considerada no balanco (m)
    P       : precipitacao (m/s)  0 neste experimento diurno
    theta_min, theta_max : limites fisicos (ponto de murcha / saturacao)
    """
    E = LE / (L * rho_agua)  # m/s (evaporacao equivalente)
    dtheta = (P - E) / D_solo * dt
    theta_novo = theta + dtheta
    theta_novo = float(np.clip(theta_novo, theta_min, theta_max))
    return theta_novo, E
