# -*- coding: utf-8 -*-
"""
MODULO 6  Fluxo de Calor Latente [LE]  Evaporacao do solo nu (Caso II)
====================================================================
Slides de referencia: "Fluxo de [LE]" e
"Fluxo de calor latente  Caso II: Evaporacao a superficie do solo umido [E]"

    LE = (rho*cp/gamma) * (h*es(T) - er) / (ra + rsoil)

    h        = e_solo / es(T)                (umidade relativa a sup. do solo)
    e_solo   = h * es(T)
    gamma    = cp*P / (0,622 * L)             (constante psicrometrica)
    rsoil    = exp[8,2 - 4,3*(theta/theta_s)] (Sellers et al., 1992)

Neste modulo:
  - es(T): pressao de vapor de saturacao (formula de Tetens, hPa)
  - h(theta): SIMPLIFICACAO DIDATICA  em vez da relacao completa com o
    potencial matricial do solo (que depende da classe textural, Clapp
    & Hornberger, 1978), usamos h = theta/theta_s (satura linearmente
    com a umidade do solo). Isso mantem o essencial do processo fisico
    (solo seco == pouca evaporacao) sem exigir parametros de solo extras.
  - rsoil(theta): formula empirica do slide (Sellers et al., 1992),
    valida como esta.
"""

import numpy as np

CP_AR = 1004.0     # J kg-1 K-1
RHO_AR = 1.2       # kg m-3
L_VAPORIZACAO = 2.5e6  # J kg-1
EPSILON_RAZAO = 0.622


def es_saturacao(T, formula="tetens"):
    """
    Pressao de vapor de saturacao es(T), em hPa.
    Formula de Tetens (T em K).
    """
    Tc = T - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    return es


def constante_psicrometrica(p=1000.0, cp=CP_AR, L=L_VAPORIZACAO, epsilon=EPSILON_RAZAO):
    """gamma = cp*P / (0,622*L)   [hPa/K, com P em hPa]"""
    return cp * p / (epsilon * L)


def umidade_relativa_solo(theta, theta_s=0.50):
    """
    h = theta/theta_s (0-1), simplificacao didatica de e_solo/es(T).
    """
    return float(np.clip(theta / theta_s, 0.0, 1.0))


def resistencia_solo(theta, theta_s=0.50):
    """
    rsoil = exp[8,2 - 4,3*(theta/theta_s)]   (Sellers et al., 1992)
    """
    return np.exp(8.2 - 4.3 * (theta / theta_s))


def fluxo_calor_latente(T, er, ra, theta, theta_s=0.50, p=1000.0,
                         rho=RHO_AR, cp=CP_AR):
    """
    Calcula LE (W/m2) e variaveis intermediarias (h, rsoil, es, gamma).
    """
    gamma = constante_psicrometrica(p=p, cp=cp)
    h = umidade_relativa_solo(theta, theta_s)
    rsoil = resistencia_solo(theta, theta_s)
    es_T = es_saturacao(T)
    LE = (rho * cp / gamma) * (h * es_T - er) / (ra + rsoil)
    return LE, {"h": h, "rsoil": rsoil, "es_T": es_T, "gamma": gamma}
