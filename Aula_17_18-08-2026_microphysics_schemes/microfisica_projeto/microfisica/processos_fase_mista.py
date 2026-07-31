# -*- coding: utf-8 -*-
"""
processos_fase_mista.py
=========================

PASSO 3 do modelo de microfisica de nuvens: adiciona NEVE (qs, Ns) e
GRAUPEL (qg, Ng), alem dos processos de interacao entre TODAS as fases
(liquida, gelo, neve, graupel) -- riming, congelamento de gotas
maiores, multiplicacao de gelo (Hallett-Mossop) e degelo.

NOTACAO DO CURSO (slides MET-756-4, Parte IV)
------------------------------------------------
    Picns   : autoconversao de gelo em neve
    Psdep   : deposicao/sublimacao sobre neve
    Pgdep   : deposicao/sublimacao sobre graupel
    Ps_sacw : neve coleta agua de nuvem (riming)      [slides: "Ps.sacw"]
    Pgacw   : graupel coleta agua de nuvem (riming)
    Ps_sacr : neve coleta chuva                        [slides: "Ps.sacr"]
    Pgacr   : graupel coleta chuva
    Piacr   : gelo coleta chuva (formando graupel)
    Pgfzr   : congelamento heterogeneo de chuva -> graupel (Bigg 1953)
    Psmlt   : degelo de neve -> chuva (T > 0 graus C)
    Pgmlt   : degelo de graupel -> chuva (T > 0 graus C)
    Pi_iacw : gelo coleta agua de nuvem                [slides: "Pi.iacw",
              adiado do Passo 2 -- ver processos_fase_gelo.py]
    Pispl   : multiplicacao de gelo por rime splintering (Hallett-Mossop)
              [adiado do Passo 2]

ESCOPO DESTE PASSO (simplificacoes deliberadas, documentadas)
-----------------------------------------------------------------
Para manter o codigo tratavel em um contexto didatico, os seguintes
termos das equacoes OFICIAIS do curso sao DELIBERADAMENTE OMITIDOS
(o graupel, neste projeto, se forma apenas por congelamento de chuva
-- Pgfzr -- e por coleta gelo-chuva -- Piacr --, nao por conversao
gradual de neve/gelo fortemente rimados):

    Picng  : conversao gelo -> graupel (por riming intenso)
    Pscng  : conversao neve -> graupel (por riming intenso)
    Pg.racs: graupel coleta neve

Essas tres conversoes dependem de um "limiar de intensidade de riming"
(criterio de Farley 1987 / Reisner et al. 1998) que exige rastrear a
densidade efetiva da particula rimada -- um refinamento que fica como
sugestao de extensao futura no README.

FORMA GERAL DE COLETA CONTINUA (usada em Ps_sacw, Pgacw, Ps_sacr,
Pgacr, Pi_iacw, Piacr)
-----------------------------------------------------------------
Todos os processos de "varredura" (uma categoria de particula maior
coleta goticulas/gotas menores em sua trajetoria de queda) usam a
mesma forma simplificada de colecao continua (aproximacao didatica,
usando o diametro medio da distribuicao em vez da integral completa
sobre os dois espectros de tamanho -- ver `colecao_continua()`):

    taxa = -E * q_alvo * N_coletor * (pi/4) * D_coletor^2 * |V_coletor - V_alvo|

Esquemas operacionais (Lin et al. 1983; Rutledge & Hobbs 1983) usam a
integral completa sobre as DUAS distribuicoes gama, o que produz
formulas fechadas mais elaboradas (ver secao 6.6 dos slides da Parte
III); aqui priorizamos a transparencia didatica sobre a precisao
operacional.
"""

import numpy as np
from .constantes import (Rv, cp, Lv, Ls, rho_i, rho_s, rho_g, rho_w, T0,
                          MU_ICE, MU_SNOW, QMIN, NMIN, gamma_func,
                          A_BIGG, B_BIGG, Ka_AR, Dv_AR,
                          QI0_AUTOCONV, C1_AUTOCONV_REF, DCS,
                          A_NEVE, B_NEVE, A_GRAUPEL, B_GRAUPEL,
                          E_IC, E_SC, E_GC, E_SR, E_GR, E_IR,
                          T_HM_MIN, T_HM_MAX, HM_SPLINTERS_POR_KG)
from .distribuicoes import lambda_gama, diametro_medio_numero, diametro_medio_massa
from .processos_fase_gelo import pressao_saturacao_gelo, razao_mistura_saturacao_gelo


# =======================================================================
# VELOCIDADES TERMINAIS DE NEVE E GRAUPEL (para coleta continua e Prprc)
# =======================================================================
def velocidade_terminal_neve(qs, Ns, rho_ar):
    """V(D)=A_NEVE*D^B_NEVE (Locatelli & Hobbs 1974). Retorna (Vq, Vn)."""
    if qs <= QMIN or Ns <= NMIN:
        return 0.0, 0.0
    lam = lambda_gama(qs, Ns, rho_ar, rho_s, MU_SNOW)
    Vq = A_NEVE * (gamma_func(MU_SNOW + 4.0 + B_NEVE) / gamma_func(MU_SNOW + 4.0)) * lam ** (-B_NEVE)
    Vn = A_NEVE * (gamma_func(MU_SNOW + 1.0 + B_NEVE) / gamma_func(MU_SNOW + 1.0)) * lam ** (-B_NEVE)
    return min(Vq, 3.0), min(Vn, 3.0)


def velocidade_terminal_graupel(qg, Ng, rho_ar):
    """V(D)=A_GRAUPEL*D^B_GRAUPEL (Rutledge & Hobbs 1984). Retorna (Vq, Vn)."""
    if qg <= QMIN or Ng <= NMIN:
        return 0.0, 0.0
    lam = lambda_gama(qg, Ng, rho_ar, rho_g, MU_SNOW)
    Vq = A_GRAUPEL * (gamma_func(MU_SNOW + 4.0 + B_GRAUPEL) / gamma_func(MU_SNOW + 4.0)) * lam ** (-B_GRAUPEL)
    Vn = A_GRAUPEL * (gamma_func(MU_SNOW + 1.0 + B_GRAUPEL) / gamma_func(MU_SNOW + 1.0)) * lam ** (-B_GRAUPEL)
    return min(Vq, 12.0), min(Vn, 12.0)


# =======================================================================
# Picns -- AUTOCONVERSAO GELO -> NEVE (Lin, Farley & Orville 1983)
# =======================================================================
def Picns(qi, Ni, T):
    """
    Picns -- autoconversao de gelo em neve: cristais de gelo crescem
    (por Pidep) ate ultrapassar um diametro/massa de corte e passam a
    ser classificados como "neve" (agregados/flocos). Nas equacoes do
    curso:

        (dp*qi/dt) contem o termo -p* Picns
        (dp*qs/dt) contem o termo +p* Picns

    Forma classica de limiar (Lin, Farley & Orville 1983):

        Picns = C1(T) * (qi - qi0),  se qi > qi0, senao 0

        C1(T) = C1_ref * exp[0.025*(T-T0)]   (mais rapido em T mais alta)

    Os novos flocos de neve formados sao assumidos com diametro Dcs
    (diametro de corte gelo->neve, MG2008), usado para converter a
    massa transferida em numero de particulas de neve criadas.

    Retorna
    -------
    dqi_dt : taxa de perda de qi (kg/kg/s, <=0)
    dNi_dt : taxa de perda de Ni (kg^-1 s^-1, <=0), assumindo fracao de
             numero igual a fracao de massa convertida
    dNs_dt : taxa de ganho de Ns (kg^-1 s^-1, >=0)
    """
    if qi <= QI0_AUTOCONV or Ni <= NMIN:
        return 0.0, 0.0, 0.0

    C1 = C1_AUTOCONV_REF * np.exp(0.025 * (T - T0))
    dqi_dt = -C1 * (qi - QI0_AUTOCONV)

    massa_media_gelo = qi / Ni
    dNi_dt = dqi_dt / massa_media_gelo  # mesma fracao de numero e massa

    massa_floco_novo = (np.pi / 6.0) * rho_s * DCS ** 3
    dNs_dt = -dqi_dt / massa_floco_novo

    return dqi_dt, dNi_dt, dNs_dt


# =======================================================================
# Psdep, Pgdep -- DEPOSICAO/SUBLIMACAO SOBRE NEVE E GRAUPEL
# =======================================================================
def _deposicao_generica(qv, qx, Nx, T, p, rho_ar, rho_x, mu_x, D_assumido_se_vazio):
    """
    Implementacao generica da equacao de crescimento por deposicao
    (mesma fisica de `Pidep`, ver `processos_fase_gelo.py` e slides
    Parte III secao 6.3.2), reutilizada para neve e graupel.
    """
    if Nx <= NMIN:
        return 0.0

    esi = pressao_saturacao_gelo(T)
    qvi = razao_mistura_saturacao_gelo(T, p)
    Swi = (qv - qvi) / qvi if qvi > 0 else 0.0

    if qx <= QMIN:
        D = D_assumido_se_vazio
    else:
        lam = lambda_gama(qx, Nx, rho_ar, rho_x, mu_x)
        D = diametro_medio_numero(lam, mu_x)

    C = D / 2.0
    Fk = (Ls / (Rv * T)) * (Ls / (Ka_AR * T) - 1.0)
    Fd = (Rv * T) / (Dv_AR(T, p) * esi)

    dmx_dt = 4.0 * np.pi * C * Swi / (Fk + Fd)
    return Nx * dmx_dt


def Psdep(qv, qs, Ns, T, p, rho_ar):
    """Psdep -- deposicao/sublimacao sobre neve (mesma fisica de Pidep)."""
    return _deposicao_generica(qv, qs, Ns, T, p, rho_ar, rho_s, MU_SNOW, DCS)


def Pgdep(qv, qg, Ng, T, p, rho_ar):
    """Pgdep -- deposicao/sublimacao sobre graupel (mesma fisica de Pidep)."""
    return _deposicao_generica(qv, qg, Ng, T, p, rho_ar, rho_g, MU_SNOW, 500.0e-6)


# =======================================================================
# COLETA CONTINUA GENERICA (riming e acrescimo entre categorias)
# =======================================================================
def colecao_continua(q_alvo, N_coletor, D_coletor, V_coletor, V_alvo, E):
    """
    Forma simplificada de colecao continua (ver docstring do modulo):

        taxa = -E * q_alvo * N_coletor * (pi/4) * D_coletor^2 * |V_coletor-V_alvo|

    Retorna a taxa de PERDA de q_alvo (kg/kg/s, <=0).
    """
    if q_alvo <= QMIN or N_coletor <= NMIN:
        return 0.0
    return -E * q_alvo * N_coletor * (np.pi / 4.0) * D_coletor ** 2 * abs(V_coletor - V_alvo)


# =======================================================================
# Pgfzr -- CONGELAMENTO HETEROGENEO DE CHUVA -> GRAUPEL (Bigg 1953)
# =======================================================================
def Pgfzr(qr, Nr, T, dt):
    """
    Pgfzr -- congelamento heterogeneo (imersao) de gotas de CHUVA
    formando graupel. Mesma fisica de `Pifzc` (ver
    `processos_fase_gelo.py`), aplicada a gotas maiores (chuva), que
    tem maior probabilidade de conter um nucleo de congelamento por
    gota (maior volume) -- por isso gotas de chuva congelam mais
    facilmente que goticulas de nuvem a mesma temperatura.

    Retorna
    -------
    dqr_dt, dNr_dt : taxas de perda de qr, Nr (kg/kg/s, kg^-1 s^-1; <=0)
    """
    if qr <= QMIN or Nr <= NMIN or T >= T0:
        return 0.0, 0.0

    massa_gota = qr / Nr
    volume_gota = massa_gota / rho_w
    taxa_por_gota = B_BIGG * np.exp(A_BIGG * (T0 - T)) * volume_gota
    taxa_por_gota = min(taxa_por_gota, 1.0 / dt)

    dNr_dt = -Nr * taxa_por_gota
    dqr_dt = -qr * taxa_por_gota
    return dqr_dt, dNr_dt


# =======================================================================
# Pmlt generico -- DEGELO (neve/graupel -> chuva, T > 0 graus C)
# =======================================================================
def Pmlt(qx, Nx, T, dt):
    """
    Degelo instantaneo de uma categoria de gelo (neve ou graupel) para
    chuva quando T > 0 graus C -- mesma aproximacao usada em `Pimlt`
    (adequada para fins didaticos; esquemas operacionais tratam o
    degelo gradual via a equacao de Rasmussen & Heymsfield 1987, slides
    Parte III, Tabela 6.6.1).

    Retorna
    -------
    dqx_dt, dNx_dt : taxas de perda de qx, Nx (kg/kg/s, kg^-1 s^-1; <=0)
    """
    if qx <= QMIN or T < T0:
        return 0.0, 0.0
    return -qx / dt, -Nx / dt


# =======================================================================
# Pispl -- MULTIPLICACAO DE GELO POR RIME SPLINTERING (Hallett-Mossop)
# =======================================================================
def Pispl(taxa_riming_total_kgkgs, T):
    """
    Pispl -- producao de novos cristais de gelo por estilhacamento
    durante o riming (mecanismo de Hallett-Mossop, ver slides Parte
    III, secao 6.8: "estilhacamento causado pelo riming"). Ativo apenas
    numa faixa estreita de temperatura, -3 graus C a -8 graus C (Mossop 1978).

    Parametros
    ----------
    taxa_riming_total_kgkgs : soma das taxas de riming (Pi_iacw+Ps_sacw+
                              Pgacw, todas em kg/kg/s, valor positivo =
                              massa de agua coletada) que ocorreram
                              neste passo de tempo.
    T : temperatura (K)

    Retorna
    -------
    dNi_dt : taxa de criacao de novos cristais de gelo (kg^-1 s^-1, >=0)
    """
    if T < T_HM_MIN or T > T_HM_MAX or taxa_riming_total_kgkgs <= 0:
        return 0.0
    return taxa_riming_total_kgkgs * HM_SPLINTERS_POR_KG
