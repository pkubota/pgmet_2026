# -*- coding: utf-8 -*-
"""
MODULO 0  Forcantes atmosfericas ("dado" nos esquemas dos slides)
====================================================================

Nos diagramas do curso (ex.: slides "Saldo de radiacao [Rn]",
"Fluxo de calor sensivel [H]", "Fluxo de calor latente"), varias
grandezas aparecem rotuladas como "(dado)": SWd, LWd, Tr, er, ur.

Isso significa que, num modelo de superficie "offline" (nao acoplado
a um modelo atmosferico), essas variaveis sao PRESCRITAS  vem de
observacoes ou, neste exercicio didatico, de funcoes analiticas que
mimetizam um ciclo diurno tipico.

Este modulo define essas funcoes de forcante em funcao da hora local
do dia (0-24h), usadas por todos os modulos seguintes.
"""

import numpy as np


def radiacao_onda_curta_incidente(hora, SWd_max=800.0, nascer=6.0, poente=18.0):
    """
    SWd(t): radiacao de onda curta incidente na superficie.

    Aproximacao classica: metade de senoide entre o nascer e o por do sol,
    zero durante a noite.

    Parametros
    ----------
    hora : float ou array
        Hora local (0-24h), pode ser fracionaria.
    SWd_max : float
        Irradiancia solar de pico ao meio-dia solar (W/m2).
    nascer, poente : float
        Horarios de nascer/por do sol (h).
    """
    hora = np.asarray(hora, dtype=float)
    frac_dia = (hora - nascer) / (poente - nascer)
    SWd = SWd_max * np.sin(np.pi * frac_dia)
    SWd = np.where((hora >= nascer) & (hora <= poente), SWd, 0.0)
    SWd = np.clip(SWd, 0.0, None)
    return SWd


def temperatura_referencia(hora, T_media=298.0, amplitude=6.0, hora_pico=15.0):
    """
    Tr(t): temperatura do ar no nivel de referencia zr (K).

    Ciclo diurno senoidal simples, com maximo em `hora_pico`
    (tipicamente 14-15h, defasado do pico solar por causa da
    inercia termica da CLP) e minimo 12h depois.
    """
    hora = np.asarray(hora, dtype=float)
    Tr = T_media + amplitude * np.cos(2.0 * np.pi * (hora - hora_pico) / 24.0)
    return Tr


def radiacao_onda_longa_incidente(Tr, emissividade_atm=0.75, sigma=5.67e-8):
    """
    LWd(t): radiacao de onda longa atmosferica incidente (W/m2).

    Aproximacao simples tipo Brutsaert/Swinbank:
        LWd = eps_atm * sigma * Tr^4
    Isso faz LWd acompanhar (com defasagem) o ciclo de Tr  mais uma
    forma pratica de fechar o balanco de radiacao sem precisar de
    dados de radiossondagem.
    """
    return emissividade_atm * sigma * Tr**4


def umidade_relativa_referencia(hora, RH_media=0.60, amplitude=0.15, hora_pico=5.0):
    """
    Umidade relativa do ar no nivel de referencia (fracao 0-1).

    Tipicamente em antifase com a temperatura (maxima de madrugada,
    minima a tarde).
    """
    hora = np.asarray(hora, dtype=float)
    RH = RH_media + amplitude * np.cos(2.0 * np.pi * (hora - hora_pico) / 24.0)
    return np.clip(RH, 0.05, 0.99)


def vento_referencia(hora, U_media=3.0, amplitude=1.0, hora_pico=14.0):
    """
    Velocidade do vento no nivel de referencia Ur (m/s).

    Pequeno ciclo diurno (mistura mecanica maior a tarde).
    """
    hora = np.asarray(hora, dtype=float)
    Ur = U_media + amplitude * np.cos(2.0 * np.pi * (hora - hora_pico) / 24.0)
    return np.clip(Ur, 0.5, None)
