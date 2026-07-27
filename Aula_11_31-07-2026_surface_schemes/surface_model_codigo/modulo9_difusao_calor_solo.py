# -*- coding: utf-8 -*-
"""
MODULO 9 - Difusao de Calor no Solo (modelo multicamadas, esquema implicito)
====================================================================
Slides de referencia: "Fluxo de calor no solo [G] - Modelo de
temperatura do solo" e "O modelo termodinamico de duas camadas de solo"

Equacao governante:

    C(theta) * dT/dt = d/dz [ K_T(theta) * dT/dz ]

Onde:
    C(theta)   = capacidade termica volumetrica do solo (J K-1 m-3)
                 = (1-theta)*C_solo + theta*C_agua      (slide, exato)
    K_T(theta) = condutividade termica do solo (W m-1 K-1)
                 fortemente dependente da umidade (Al Nakshabandi e
                 Kohnke, 1965), via o "Pf" (log do potencial matricial):

    Pf = log10[ |psi_s| * (theta_s/theta)^b ]     (psi_s em cm)
    kappa(theta) = 420*exp(-Pf+2.7)  se Pf <= 5.1     (difusividade termica)
                 = 0.1722            se Pf  > 5.1

    K_T(theta) = kappa(theta) * C(theta)

NOTA DIDATICA: a formula empirica de Al Nakshabandi e Kohnke (1965)
citada no slide fornece originalmente uma DIFUSIVIDADE termica em
unidades de 10^-3 cm2/s (ver a figura "Difusividades" do slide, eixo
em m2/s ~1e-7 a 1e-8). Aqui usamos um fator de escala explicito
(`fator_escala`) para converter o valor bruto da formula para m2/s
dentro da faixa fisica esperada para solos (kappa ~ 1e-9 a 1e-5 m2/s).
Se for calibrar o modelo com dados reais, ajuste `fator_escala`.

Discretizacao no tempo: esquema IMPLICITO (backward Euler), conforme
o slide ("esquema totalmente implicito"). O sistema linear resultante
e tridiagonal e resolvido pelo ALGORITMO DE THOMAS (equivalente a
inversao da matriz M do slide, mas sem montar a matriz cheia).

Contorno superior: temperatura da superficie T_s conhecida (Dirichlet),
obtida do balanco de energia da superficie (ver script principal).
Contorno inferior: temperatura constante especificada T_fundo (Dirichlet).
"""

import numpy as np

from modulo8_difusao_umidade_solo import obter_parametros_solo


def capacidade_termica(theta, C_solo=1.26e6, C_agua=4.2e6):
    """C(theta) = (1-theta)*C_solo + theta*C_agua   [J K-1 m-3]"""
    return (1.0 - theta) * C_solo + theta * C_agua


def pf_potencial_matricial(theta, psi_s, theta_s, b):
    """
    Pf = log10[ |psi_s(cm)| * (theta_s/theta)^b ]
    psi_s fornecido em metros (tabela Clapp-Hornberger) -> convertido p/ cm.
    """
    theta_eff = np.clip(theta, 1.0e-3, theta_s * 0.999)
    psi_s_cm = abs(psi_s) * 100.0
    Pf = np.log10(psi_s_cm * (theta_s / theta_eff) ** b)
    return Pf


def difusividade_termica(theta, psi_s, theta_s, b, fator_escala=1.0e-8):
    """
    kappa(theta) [m2/s], formula de Al Nakshabandi e Kohnke (1965),
    reescalada por `fator_escala` (ver nota didatica no cabecalho).
    """
    Pf = pf_potencial_matricial(theta, psi_s, theta_s, b)
    kappa_raw = np.where(Pf <= 5.1, 420.0 * np.exp(-Pf + 2.7), 0.1722)
    return kappa_raw * fator_escala


def condutividade_termica(theta, psi_s, theta_s, b,
                           C_solo=1.26e6, C_agua=4.2e6, fator_escala=1.0e-8,
                           K_T_min=0.15, K_T_max=0.7):
    """
    K_T(theta) = kappa(theta) * C(theta)   [W m-1 K-1]

    NOTA DIDATICA (calibracao): a formula empirica de Al Nakshabandi e
    Kohnke (1965) produz uma faixa dinamica MUITO maior do que a
    observada em solos reais (K_T real varia tipicamente entre ~0,15
    W/mK em solo seco e ~2,5 W/mK em solo saturado, um fator ~15; a
    formula bruta, sem calibracao de site, pode gerar fatores >1000).
    Por isso o resultado e limitado (clip) ao intervalo fisico
    [K_T_min, K_T_max]. Isso preserva o COMPORTAMENTO qualitativo
    ensinado no slide (K_T cresce fortemente com theta, solo seco
    isola termicamente) sem gerar fluxos de solo (G) irrealistas.

    O valor padrao de K_T_max (0,7 W/mK) foi escolhido para reproduzir,
    no modelo de SOLO NU (sem vegetacao/sombreamento), uma particao
    G/Rn no pico solar em torno de 0,3 - consistente com observacoes
    de campo em solo descoberto (tipicamente 0,2-0,4; mais alta do
    que em superficies vegetadas, que tem o dossel interceptando
    radiacao antes de chegar ao solo). Se for representar um solo
    mais condutor (ex.: bem saturado, textura arenosa compactada),
    aumente K_T_max; para solo mais seco/organico, reduza.
    """
    kappa = difusividade_termica(theta, psi_s, theta_s, b, fator_escala)
    C = capacidade_termica(theta, C_solo, C_agua)
    K_T = kappa * C
    return np.clip(K_T, K_T_min, K_T_max)


def resolver_tridiagonal(a, b, c, d):
    """
    Algoritmo de Thomas para sistema tridiagonal A*x = d, com:
        a : subdiagonal (a[0] nao usado)
        b : diagonal principal
        c : superdiagonal (c[-1] nao usado)
        d : lado direito
    Todos os arrays com o mesmo tamanho n.
    """
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def difusao_calor_solo_implicito(T, theta, dz, dt, T_superficie, T_fundo,
                                  classe_solo="franco_argiloso",
                                  C_solo=1.26e6, C_agua=4.2e6,
                                  fator_escala_kt=1.0e-8):
    """
    Avanca o perfil de temperatura do solo (multicamadas) por um passo
    dt, com contorno Dirichlet no topo (T_superficie) e na base (T_fundo).

    Parametros
    ----------
    T             : array (n,) - temperatura de cada camada (K)
    theta         : array (n,) - umidade volumetrica de cada camada (m3/m3)
    dz            : array (n,) - espessura de cada camada (m)
    dt            : passo de tempo (s)
    T_superficie  : temperatura da superficie no passo n+1 (K) - Dirichlet
    T_fundo       : temperatura do fundo do perfil (K) - Dirichlet

    Retorna
    -------
    T_novo : array (n,)
    K_T    : array (n,) - condutividade termica usada (diagnostico)
    C      : array (n,) - capacidade termica usada (diagnostico)
    """
    psi_s, theta_s, b, Ks = obter_parametros_solo(classe_solo)

    n = len(T)
    dz = np.asarray(dz, dtype=float)
    T = np.asarray(T, dtype=float)
    theta = np.asarray(theta, dtype=float)

    C = capacidade_termica(theta, C_solo, C_agua)
    K_T = condutividade_termica(theta, psi_s, theta_s, b, C_solo, C_agua, fator_escala_kt)

    # distancias "efetivas" entre pontos onde T e definida
    dz_tilde = np.zeros(n + 1)
    dz_tilde[0] = 0.5 * dz[0]                      # superficie -> centro camada 0
    dz_tilde[1:n] = 0.5 * (dz[:-1] + dz[1:])        # entre centros de camadas
    dz_tilde[n] = 0.5 * dz[-1]                      # centro camada n-1 -> fundo

    K_T_interf = np.zeros(n + 1)
    K_T_interf[0] = K_T[0]
    K_T_interf[1:n] = 0.5 * (K_T[:-1] + K_T[1:])
    K_T_interf[n] = K_T[-1]

    A = np.zeros(n)   # ligacao com camada k-1 (ou superficie se k=0)
    Cc = np.zeros(n)  # ligacao com camada k+1 (ou fundo se k=n-1)
    for k in range(n):
        A[k] = dt / (C[k] * dz[k]) * K_T_interf[k] / dz_tilde[k]
        Cc[k] = dt / (C[k] * dz[k]) * K_T_interf[k + 1] / dz_tilde[k + 1]
    B = 1.0 + A + Cc

    a = np.zeros(n)
    b_diag = B.copy()
    c = np.zeros(n)
    d = T.copy()

    a[1:] = -A[1:]
    c[:-1] = -Cc[:-1]
    d[0] += A[0] * T_superficie
    d[-1] += Cc[-1] * T_fundo

    T_novo = resolver_tridiagonal(a, b_diag, c, d)
    return T_novo, K_T, C
