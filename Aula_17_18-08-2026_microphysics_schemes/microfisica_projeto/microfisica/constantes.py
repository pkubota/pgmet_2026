# -*- coding: utf-8 -*-
"""
constantes.py
=============

Constantes fisicas e parametros do esquema de microfisica de nuvens.

Este modulo reune todas as constantes usadas nas parametrizacoes de
microfisica implementadas ao longo da disciplina (PASSO 1: chuva quente;
PASSO 2: fase gelo; PASSO 3: interacoes de fase mista), seguindo a
notacao usada em:

    Morrison, H., & Gettelman, A. (2008): A New Two-Moment Bulk
    Stratiform Cloud Microphysics Scheme in CAM3, Part I. J. Climate, 21.
    (doi:10.1175/2008JCLI2105.1)  -> referido aqui como "MG2008"

    Khairoutdinov, M. F., & Kogan, Y. (2000): A new cloud physics
    parameterization in a large-eddy simulation model of marine
    stratocumulus. Mon. Wea. Rev., 128. -> "KK2000"

    Thompson, G., Rasmussen, R. M., & Manning, K. (2004): Explicit
    forecasts of winter precipitation using an improved bulk microphysics
    scheme. Part I. Mon. Wea. Rev., 132. -> "Thompson2004/2008"

Todas as unidades sao SI, exceto onde explicitamente indicado (algumas
formulas empiricas de autoconversao/acrecao exigem unidades especificas
-- isso e tratado localmente dentro de cada funcao de processo, nunca
aqui).
"""

import numpy as np

# ---------------------------------------------------------------------
# 1) Constantes termodinamicas do ar seco e do vapor d'agua
# ---------------------------------------------------------------------
Rd = 287.05          # J kg^-1 K^-1  -- constante do gas para o ar seco
Rv = 461.5           # J kg^-1 K^-1  -- constante do gas para o vapor d'agua
cp = 1005.0          # J kg^-1 K^-1  -- calor especifico do ar seco a p cte
g = 9.81             # m s^-2        -- aceleracao da gravidade

Lv = 2.501e6         # J kg^-1  -- calor latente de vaporizacao (0 C)
Ls = 2.834e6         # J kg^-1  -- calor latente de sublimacao (0 C)
Lf = Ls - Lv         # J kg^-1  -- calor latente de fusao (~3.33e5 J/kg)

T0 = 273.15          # K  -- ponto de fusao do gelo (0 C)

# ---------------------------------------------------------------------
# 2) Densidades das especies de hidrometeoros (kg m^-3)
# ---------------------------------------------------------------------
rho_w = 1000.0       # densidade da agua liquida
rho_i = 500.0        # densidade "bulk" assumida para o gelo de nuvem
rho_s = 100.0        # densidade "bulk" assumida para a neve (baixa, agregados)
rho_g = 400.0        # densidade "bulk" assumida para graupel/granizo mole

# ---------------------------------------------------------------------
# 3) Parametros das distribuicoes gama assumidas para cada categoria
#
# Assume-se que o espectro de tamanhos de cada hidrometeoro segue uma
# distribuicao gama:
#
#       N(D) = N0 * D^mu * exp(-lambda * D)
#
# onde D e o diametro (m), N0 e o parametro de intercepto, mu e o
# parametro de forma (shape) e lambda e o parametro de inclinacao
# (slope). mu=0 recupera a distribuicao exponencial de Marshall-Palmer.
#
# Aqui fixamos mu por categoria (valor simplificado / didatico).
# O MG2008 usa uma relacao de Nc para mu no caso das goticulas de nuvem
# (Martin et al. 1994); aqui simplificamos com um valor fixo para
# facilitar o entendimento dos processos no Passo 1.
# ---------------------------------------------------------------------
MU_CLOUD = 1.0       # forma da distribuicao de goticulas de nuvem (qc, Nc)
MU_RAIN = 0.0        # forma da distribuicao de gotas de chuva (qr, Nr) -> exponencial
MU_ICE = 0.0         # forma da distribuicao de cristais de gelo (Passo 2)
MU_SNOW = 0.0        # forma da distribuicao de neve (Passo 2/3)

# ---------------------------------------------------------------------
# 4) Limites fisicos e parametros de processo
# ---------------------------------------------------------------------
QMIN = 1.0e-12       # kg/kg -- limiar minimo de razao de mistura para considerar "ha hidrometeoro"
NMIN = 1.0e-6        # kg^-1 -- limiar minimo de concentracao numerica

TAU_AUTO_ICE = 180.0     # s  -- escala de tempo da autoconversao gelo->neve (3 min, MG2008 eq. 29-30)
DCS = 200.0e-6           # m  -- diametro de corte cristal de gelo -> neve (Dcs, MG2008)

# Eficiencias de coleta (adimensional), ver MG2008 secao "g. Other collection processes"
E_CI_SNOW = 0.1      # eficiencia de coleta gelo-neve (Reisner et al. 1998)
E_RAIN_SNOW = 1.0    # eficiencia de coleta chuva-neve em condicoes subfreezing (Ikawa e Saito 1990)

# ---------------------------------------------------------------------
# 5) PASSO 2: nucleacao primaria de gelo, congelamento e propriedades
#    de transporte de vapor (usadas em Pidsn, Pidep, Pifzc)
# ---------------------------------------------------------------------
# Nucleacao primaria de gelo (Cooper 1986), ver MG2008 eq. (20) e slides
# da Parte III (modos de nucleacao: deposicao, condensacao, imersao,
# contato -- Vali 1999):
NIN0_COOPER = 0.005      # L^-1, prefator da formula de Cooper (1986)
NIN_B_COOPER = 0.304     # K^-1, coeficiente exponencial
T_NUCLEACAO_MAX = 268.15  # K (-5 graus C) -- nucleacao so ocorre para T < este valor
TAU_ATIVACAO = 1200.0    # s (20 min) -- escala de tempo de ativacao (MG2008)
D_GELO_NUCLEADO = 10.0e-6  # m -- diametro assumido de um cristal recem-nucleado

# Congelamento por imersao (Bigg 1953), forma usada em Reisner et al.
# (1998, RRB) e citada nos slides da Parte III (secao "Congelamento
# heterogeneo -- imersao"):
A_BIGG = 0.66            # K^-1
B_BIGG = 1.0e8           # m^-3 s^-1 (equivalente a 100 cm^-3 s^-1)
T_HOMOGENEO = T0 - 40.0  # K (-40 graus C) -- congelamento homogeneo instantaneo abaixo disto

# Propriedades de transporte de vapor (usadas na equacao de crescimento
# por deposicao, slides Parte III, secao 6.3.2, Pruppacher & Klett 1997)
Ka_AR = 2.4e-2           # W m^-1 K^-1 -- condutividade termica do ar
def Dv_AR(T, p):
    """
    Difusividade do vapor d'agua no ar (m^2/s), dependente de T e p
    (formula aproximada de Hall & Pruppacher 1976, amplamente usada em
    esquemas de microfisica):

        Dv = 2.11e-5 * (T/273.15)^1.94 * (101325/p)   [m^2/s]
    """
    return 2.11e-5 * (T / 273.15) ** 1.94 * (101325.0 / p)

# ---------------------------------------------------------------------
# 6) PASSO 3: neve, graupel e interacoes de fase mista
# ---------------------------------------------------------------------
# Autoconversao gelo -> neve (Lin, Farley & Orville 1983, forma classica
# de limiar): Picns = C1(T) * (qi - qi0), se qi > qi0
QI0_AUTOCONV = 1.0e-3    # kg/kg -- limiar de massa de gelo para autoconversao
C1_AUTOCONV_REF = 1.0e-3  # s^-1  -- taxa de referencia a 0 graus C

# Relacoes potencia V(D)=a*D^b para velocidade terminal (Lin et al. 1983;
# Locatelli & Hobbs 1974 para neve; Rutledge & Hobbs 1984 para graupel)
A_NEVE, B_NEVE = 11.72, 0.41        # neve (agregados)
A_GRAUPEL, B_GRAUPEL = 19.3, 0.37   # graupel/granizo mole

# Eficiencias de coleta (riming e acrescimo), adimensionais
E_IC = 0.5     # gelo coleta agua de nuvem (Pi_iacw)
E_SC = 1.0     # neve coleta agua de nuvem (Ps_sacw), eficiente pois ja e > 0 graus C-adjacent
E_GC = 1.0     # graupel coleta agua de nuvem (Pgacw)
E_SR = 1.0     # neve coleta chuva (Ps_sacr)
E_GR = 1.0     # graupel coleta chuva (Pgacr)
E_IR = 1.0     # gelo coleta chuva -> graupel (Piacr)

# Hallett-Mossop (Mossop 1978): faixa de temperatura ativa e produtividade
T_HM_MIN = T0 - 8.0   # -8 graus C
T_HM_MAX = T0 - 3.0   # -3 graus C
HM_SPLINTERS_POR_KG = 3.5e14  # no de estilhacos produzidos por kg de agua rimada


def gamma_func(x):
    """Funcao gama (wrapper conveniente em torno de scipy)."""
    from scipy.special import gamma
    return gamma(x)
