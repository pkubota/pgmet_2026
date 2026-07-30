# -*- coding: utf-8 -*-
"""
=====================================================================
 Tc, Tg, Td COMO SISTEMA TRIDIAGONAL - Algoritmo de Thomas
 MET-576-4 - Modelagem Numerica da Atmosfera - Dr. Paulo Yoshio Kubota
=====================================================================

Pergunta respondida por este script: "para resolver Tc, Tg e Td nao se
resolve com uma matriz tridiagonal?"

RESPOSTA CURTA: sim, quando as TRES temperaturas (Tc=dossel, Tg=solo
superficial, Td=solo profundo) sao tratadas como incognitas SIMULTANEAS
de um unico sistema implicito, a matriz resultante e TRIDIAGONAL, pois:
  - Tc so troca energia diretamente com Tg (nao com Td)      -> K(1,3)=0
  - Td so troca energia diretamente com Tg (nao com Tc)      -> K(3,1)=0
  - Tg e o elo do meio, acoplado aos dois vizinhos            -> K(2,1),K(2,3) != 0
Essa estrutura "em cadeia" Tc-Tg-Td e exatamente uma matriz tridiagonal,
e o metodo eficiente para resolve-la e o ALGORITMO DE THOMAS (eliminacao
progressiva + substituicao regressiva), custo O(n), em vez de eliminacao
de Gauss/inversao geral de matriz, custo O(n^3). A diferenca de custo e
irrelevante para n=3, mas e o motivo pelo qual modelos de solo com MUITAS
camadas (Tg, Td1, Td2, ..., TdN) sempre usam solver tridiagonal.

RESSALVAS IMPORTANTES (comparando com o material da disciplina):

1) O codigo Fortran operacional do SSiB (module_ssibsub.F90, subrotina
   TEMRS1) NAO resolve o sistema 3x3 completo. Ele resolve implicitamente
   apenas o par (Tc,Tg) - um sistema 2x2, com
       DENOM = CCODTC*GCODTG - CCODTG*GCODTC
   - e atualiza Td SEPARADAMENTE, de forma EXPLICITA (force-restore, na
   subrotina UPDAT1). Isso e um atalho comum em LSMs, pois Td varia numa
   escala de tempo muito mais lenta (dias) que Tc e Tg (minutos/horas), o
   que torna o passo explicito estavel mesmo com dt=1h.

2) O modelo didatico simplificado usado nas demais entregas desta
   disciplina (baseado no vegetacao.pdf) vai um passo alem: junta Tc e Tg
   numa unica temperatura de "pele" T (dossel+solo), entao la nem existe
   um sistema 3x3 - e apenas T (implicito, escalar) e Td (explicito).

Este script aqui implementa a versao MAIS COMPLETA (Tc, Tg, Td como tres
incognitas simultaneas), reaproveitando a fisica de Tc/Tg com a rede de
tres resistencias (ra,rb,rd) do primeiro material da disciplina (Parte 3
- Surface Model Vegetation), e resolve o sistema de tres formas
diferentes para comparacao:

  (A) Algoritmo de THOMAS (tridiagonal, O(n))
  (B) Matriz CHEIA 3x3 (numpy.linalg.solve, Gauss generico, O(n^3))
  (C) Atalho do SSiB: 2x2 implicito (Tc,Tg) + Td EXPLICITO (force-restore)

(A) e (B) devem dar EXATAMENTE o mesmo resultado (validando o solver
tridiagonal). (C) e o que o codigo operacional realmente faz, e deve
ficar proximo de (A)/(B) na maior parte do tempo, mas pode divergir um
pouco quando Td muda mais rapido que o usual (ex.: dt grande).
=====================================================================
"""

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTES FISICAS
# ---------------------------------------------------------------------------
SIGMA  = 5.67e-8
CPAIR  = 1004.0
RHOAIR = 1.20
HLAT   = 2.50e6
PSUR   = 1000.0e2
TF     = 273.15
TIMCON = np.pi / 86400.0     # constante de force-restore (periodo de 1 dia)


def esat(T):
    Tc = T - TF
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def desat_dT(T):
    Tc = T - TF
    es = esat(T)
    return es * 17.67 * 243.5 / (Tc + 243.5) ** 2


# ---------------------------------------------------------------------------
# ALGORITMO DE THOMAS (solver tridiagonal generico, O(n))
# ---------------------------------------------------------------------------
def thomas(a, b, c, d):
    """
    Resolve o sistema tridiagonal:
        a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]     (a[0] e c[n-1] nao usados)
    Custo O(n): uma passada progressiva (elimina a subdiagonal) e uma
    passada regressiva (substituicao). Generaliza para qualquer numero de
    camadas (ex.: Tc, Tg, Td1, Td2, ..., TdN em um modelo de solo com N
    camadas), nao apenas para n=3.
    """
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m

    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------
# PARAMETROS DO MODELO (Tc, Tg, Td com rede de 3 resistencias ra, rb, rd)
# ---------------------------------------------------------------------------
class Parametros:
    def __init__(self):
        self.LAI = 3.0
        self.VCOVER = 0.9
        self.K_EXT = 0.5
        self.ALB_C = 0.20
        self.ALB_G = 0.15
        self.EPS_C = 0.96
        self.EPS_G = 0.94
        self.RST_MIN = 100.0
        self.RST_MAX = 5000.0
        self.RG_REF = 100.0

        self.CCX = 3.0e3     # capacidade termica do dossel [J/m2/K]
        self.CG = 2.0e5      # capacidade termica da superficie do solo [J/m2/K]
        self.CD = 1.0e7      # capacidade termica do solo profundo [J/m2/K]

        self.RA0 = 60.0
        self.RB0 = 20.0
        self.RD0 = 90.0
        self.U_MIN = 0.5

        # umidade/resistencia do solo mantidas simples e FIXAS neste script
        # (o foco aqui e o metodo numerico de solucao de Tc,Tg,Td, nao o
        # balanco hidrico - que ja foi tratado nos outros scripts)
        self.Wsoil_fixo = 0.55
        self.RSOIL_MIN = 100.0
        self.Wc = 0.0
        self.Wg = 0.0


def gerar_forcante(nhoras=72, dt=3600.0, seed=42):
    rng = np.random.default_rng(seed)
    n = int(nhoras * 3600 / dt) + 1
    horas = np.arange(n) * dt / 3600.0
    hdia = horas % 24.0

    Rg = 850.0 * np.clip(np.sin(np.pi * (hdia - 6.0) / 12.0), 0, None)
    Rg += rng.normal(0, 5, n) * (Rg > 0)
    Rg = np.clip(Rg, 0, None)

    Ta = 296.0 + 6.0 * np.sin(2 * np.pi * (hdia - 9.0) / 24.0) + rng.normal(0, 0.2, n)
    RH = np.clip(0.55 + 0.30 * np.sin(2 * np.pi * (hdia - 21.0) / 24.0), 0.25, 0.98)
    qa = 0.622 * (RH * esat(Ta)) / PSUR
    U = np.clip(2.0 + 1.5 * np.clip(np.sin(np.pi * (hdia - 6.0) / 12.0), 0, None)
                + rng.normal(0, 0.15, n), 0.3, None)
    eps_atm = np.clip(0.70 + 0.20 * (RH * esat(Ta) / 1000.0) ** 0.25, 0.6, 0.98)
    Ld = eps_atm * SIGMA * Ta ** 4

    return dict(horas=horas, dt=dt, Rg=Rg, Ta=Ta, qa=qa, U=U, Ld=Ld)


# ---------------------------------------------------------------------------
# CALCULO DOS FLUXOS Tc/Tg (rede ra,rb,rd) + DERIVADAS - comum aos 3 metodos
# ---------------------------------------------------------------------------
def calcular_fluxos_tc_tg(Tc, Tg, forc_i, p):
    Rg, Ta, qa, U, Ld = forc_i
    U = max(U, p.U_MIN)

    ra = p.RA0 / U
    rb = p.RB0 / np.sqrt(U)
    rd = p.RD0 / U
    D1 = 1.0 / ra + 1.0 / rb + 1.0 / rd

    thermk = np.exp(-p.K_EXT * p.LAI)
    fc_abs = p.VCOVER * (1.0 - thermk)
    Rsw_c = Rg * (1.0 - p.ALB_C) * fc_abs
    Rsw_g = Rg * (1.0 - p.ALB_G) * (1.0 - fc_abs)

    f_solar = Rg / (Rg + p.RG_REF) if Rg > 0 else 0.0
    f_solo = p.Wsoil_fixo
    rst = p.RST_MIN / max(f_solar * f_solo, 1e-3)
    rst = min(rst, p.RST_MAX) if Rg > 0 else p.RST_MAX
    rsoil = p.RSOIL_MIN / p.Wsoil_fixo ** 0.5
    hr = np.clip(p.Wsoil_fixo ** 0.2, 0.10, 1.0)

    rcc = rst + 2.0 * rb
    Cc_cond = (1.0 - p.Wc) / rcc + p.Wc / (2.0 * rb)
    Cg_cond = (1.0 - p.Wg) * hr / (rsoil + rd) + p.Wg / rd
    Dq = 1.0 / ra + Cc_cond + Cg_cond

    dTac_dTc = (1.0 / rb) / D1
    dTac_dTg = (1.0 / rd) / D1

    esTc, esTg = esat(Tc), esat(Tg)
    qsTc = 0.622 * esTc / PSUR
    qsTg = 0.622 * esTg / PSUR
    dqsTc = 0.622 * desat_dT(Tc) / PSUR
    dqsTg = 0.622 * desat_dT(Tg) / PSUR

    Tac0 = (Tg / rd + Tc / rb + Ta / ra) / D1
    qac0 = (Cc_cond * qsTc + Cg_cond * qsTg + qa / ra) / Dq

    Lc0 = p.EPS_C * (Ld + p.VCOVER * SIGMA * Tg ** 4 - 2.0 * SIGMA * Tc ** 4)
    Lg0 = p.EPS_G * (thermk * Ld + (1.0 - thermk) * p.VCOVER * SIGMA * Tc ** 4 - SIGMA * Tg ** 4)
    Rn_c0 = Rsw_c + Lc0
    Rn_g0 = Rsw_g + Lg0
    dRnc_dTc = p.EPS_C * (-8.0 * SIGMA * Tc ** 3)
    dRnc_dTg = p.EPS_C * (4.0 * p.VCOVER * SIGMA * Tg ** 3)
    dRng_dTg = p.EPS_G * (-4.0 * SIGMA * Tg ** 3)
    dRng_dTc = p.EPS_G * (4.0 * (1.0 - thermk) * p.VCOVER * SIGMA * Tc ** 3)

    Hc0 = RHOAIR * CPAIR * (Tc - Tac0) / rb
    Hg0 = RHOAIR * CPAIR * (Tg - Tac0) / rd
    dHc_dTc = RHOAIR * CPAIR / rb * (1.0 - dTac_dTc)
    dHc_dTg = -RHOAIR * CPAIR / rb * dTac_dTg
    dHg_dTg = RHOAIR * CPAIR / rd * (1.0 - dTac_dTg)
    dHg_dTc = -RHOAIR * CPAIR / rd * dTac_dTc

    dqac_dTc = Cc_cond * dqsTc / Dq
    dqac_dTg = Cg_cond * dqsTg / Dq
    Ec0 = RHOAIR * HLAT * Cc_cond * (qsTc - qac0)
    Eg0 = RHOAIR * HLAT * Cg_cond * (qsTg - qac0)
    dEc_dTc = RHOAIR * HLAT * Cc_cond * (dqsTc - dqac_dTc)
    dEc_dTg = RHOAIR * HLAT * Cc_cond * (-dqac_dTg)
    dEg_dTg = RHOAIR * HLAT * Cg_cond * (dqsTg - dqac_dTg)
    dEg_dTc = RHOAIR * HLAT * Cg_cond * (-dqac_dTc)

    return dict(Rn_c0=Rn_c0, Rn_g0=Rn_g0, dRnc_dTc=dRnc_dTc, dRnc_dTg=dRnc_dTg,
                dRng_dTg=dRng_dTg, dRng_dTc=dRng_dTc,
                Hc0=Hc0, Hg0=Hg0, dHc_dTc=dHc_dTc, dHc_dTg=dHc_dTg,
                dHg_dTg=dHg_dTg, dHg_dTc=dHg_dTc,
                Ec0=Ec0, Eg0=Eg0, dEc_dTc=dEc_dTc, dEc_dTg=dEc_dTg,
                dEg_dTg=dEg_dTg, dEg_dTc=dEg_dTc)


# ---------------------------------------------------------------------------
# (A) SISTEMA TRIDIAGONAL 3x3 - Algoritmo de Thomas
# ---------------------------------------------------------------------------
def passo_tridiagonal(state, forc_i, p, dt):
    """
    Monta o sistema tridiagonal para (DTc, DTg, DTd) e resolve com o
    algoritmo de Thomas. Estrutura (a=subdiagonal, b=diagonal, c=
    superdiagonal):

        linha 1 (Tc):  b1*DTc + c1*DTg + 0    *DTd = d1
        linha 2 (Tg):  a2*DTc + b2*DTg + c2   *DTd = d2
        linha 3 (Td):  0*DTc  + a3*DTg + b3   *DTd = d3

    Note a[0]=0 (Tc nao depende de Td) e c[2]=0 (Td nao depende de Tc):
    exatamente a assinatura de uma matriz tridiagonal.
    """
    Tc, Tg, Td = state
    fx = calcular_fluxos_tc_tg(Tc, Tg, forc_i, p)

    CCODTC = p.CCX / dt - fx['dRnc_dTc'] + fx['dHc_dTc'] + fx['dEc_dTc']
    CCODTG = -fx['dRnc_dTg'] + fx['dHc_dTg'] + fx['dEc_dTg']
    CCORHS = fx['Rn_c0'] - fx['Hc0'] - fx['Ec0']

    GCODTC = -fx['dRng_dTc'] + fx['dHg_dTc'] + fx['dEg_dTc']
    GCODTG = p.CG / dt + TIMCON * p.CG * 2.0 - fx['dRng_dTg'] + fx['dHg_dTg'] + fx['dEg_dTg']
    GCODTD = -TIMCON * p.CG * 2.0
    GCORHS = fx['Rn_g0'] - TIMCON * p.CG * 2.0 * (Tg - Td) - fx['Hg0'] - fx['Eg0']

    # linha 3: Cd*dTd/dt = TIMCON*Cg*2*(Tg-Td)  ->  linearizada em DTg, DTd
    DCODTG = -TIMCON * p.CG * 2.0
    DCODTD = p.CD / dt + TIMCON * p.CG * 2.0
    DCORHS = TIMCON * p.CG * 2.0 * (Tg - Td)

    a = np.array([0.0, GCODTC, DCODTG])
    b = np.array([CCODTC, GCODTG, DCODTD])
    c = np.array([CCODTG, GCODTD, 0.0])
    d = np.array([CCORHS, GCORHS, DCORHS])

    DTc, DTg, DTd = thomas(a, b, c, d)

    Rn_c = fx['Rn_c0'] + fx['dRnc_dTc'] * DTc + fx['dRnc_dTg'] * DTg
    Rn_g = fx['Rn_g0'] + fx['dRng_dTc'] * DTc + fx['dRng_dTg'] * DTg
    Hc = fx['Hc0'] + fx['dHc_dTc'] * DTc + fx['dHc_dTg'] * DTg
    Hg = fx['Hg0'] + fx['dHg_dTc'] * DTc + fx['dHg_dTg'] * DTg
    Ec = fx['Ec0'] + fx['dEc_dTc'] * DTc + fx['dEc_dTg'] * DTg
    Eg = fx['Eg0'] + fx['dEg_dTc'] * DTc + fx['dEg_dTg'] * DTg
    G = TIMCON * p.CG * 2.0 * (Tg + DTg - Td)

    novo_estado = np.array([Tc + DTc, Tg + DTg, Td + DTd])
    diag = dict(Rn_c=Rn_c, Rn_g=Rn_g, Hc=Hc, Hg=Hg, Ec=Ec, Eg=Eg, G=G)
    return novo_estado, diag


# ---------------------------------------------------------------------------
# (B) MATRIZ CHEIA 3x3 - numpy.linalg.solve (para validar o solver tridiagonal)
# ---------------------------------------------------------------------------
def passo_matriz_cheia(state, forc_i, p, dt):
    """Mesmo sistema linear de passo_tridiagonal(), mas resolvido por
    inversao/eliminacao de Gauss GENERICA (numpy.linalg.solve), sem
    explorar a estrutura tridiagonal. Deve dar o MESMO resultado que o
    algoritmo de Thomas - serve para validar a implementacao."""
    Tc, Tg, Td = state
    fx = calcular_fluxos_tc_tg(Tc, Tg, forc_i, p)

    CCODTC = p.CCX / dt - fx['dRnc_dTc'] + fx['dHc_dTc'] + fx['dEc_dTc']
    CCODTG = -fx['dRnc_dTg'] + fx['dHc_dTg'] + fx['dEc_dTg']
    CCORHS = fx['Rn_c0'] - fx['Hc0'] - fx['Ec0']

    GCODTC = -fx['dRng_dTc'] + fx['dHg_dTc'] + fx['dEg_dTc']
    GCODTG = p.CG / dt + TIMCON * p.CG * 2.0 - fx['dRng_dTg'] + fx['dHg_dTg'] + fx['dEg_dTg']
    GCODTD = -TIMCON * p.CG * 2.0
    GCORHS = fx['Rn_g0'] - TIMCON * p.CG * 2.0 * (Tg - Td) - fx['Hg0'] - fx['Eg0']

    DCODTG = -TIMCON * p.CG * 2.0
    DCODTD = p.CD / dt + TIMCON * p.CG * 2.0
    DCORHS = TIMCON * p.CG * 2.0 * (Tg - Td)

    K = np.array([[CCODTC, CCODTG, 0.0],
                  [GCODTC, GCODTG, GCODTD],
                  [0.0,    DCODTG, DCODTD]])
    Y = np.array([CCORHS, GCORHS, DCORHS])

    DTc, DTg, DTd = np.linalg.solve(K, Y)

    Rn_c = fx['Rn_c0'] + fx['dRnc_dTc'] * DTc + fx['dRnc_dTg'] * DTg
    Rn_g = fx['Rn_g0'] + fx['dRng_dTc'] * DTc + fx['dRng_dTg'] * DTg
    Hc = fx['Hc0'] + fx['dHc_dTc'] * DTc + fx['dHc_dTg'] * DTg
    Hg = fx['Hg0'] + fx['dHg_dTc'] * DTc + fx['dHg_dTg'] * DTg
    Ec = fx['Ec0'] + fx['dEc_dTc'] * DTc + fx['dEc_dTg'] * DTg
    Eg = fx['Eg0'] + fx['dEg_dTc'] * DTc + fx['dEg_dTg'] * DTg
    G = TIMCON * p.CG * 2.0 * (Tg + DTg - Td)

    novo_estado = np.array([Tc + DTc, Tg + DTg, Td + DTd])
    diag = dict(Rn_c=Rn_c, Rn_g=Rn_g, Hc=Hc, Hg=Hg, Ec=Ec, Eg=Eg, G=G)
    return novo_estado, diag


# ---------------------------------------------------------------------------
# (C) ATALHO DO SSIB OPERACIONAL: 2x2 implicito (Tc,Tg) + Td EXPLICITO
# ---------------------------------------------------------------------------
def passo_ssib_2x2_mais_explicito(state, forc_i, p, dt):
    """Reproduz o que o codigo Fortran realmente faz (module_ssibsub.F90,
    TEMRS1 + UPDAT1): resolve implicitamente APENAS (Tc,Tg) - sistema 2x2 -
    e atualiza Td em seguida de forma EXPLICITA (force-restore simples)."""
    Tc, Tg, Td = state
    fx = calcular_fluxos_tc_tg(Tc, Tg, forc_i, p)

    CCODTC = p.CCX / dt - fx['dRnc_dTc'] + fx['dHc_dTc'] + fx['dEc_dTc']
    CCODTG = -fx['dRnc_dTg'] + fx['dHc_dTg'] + fx['dEc_dTg']
    CCORHS = fx['Rn_c0'] - fx['Hc0'] - fx['Ec0']

    GCODTC = -fx['dRng_dTc'] + fx['dHg_dTc'] + fx['dEg_dTc']
    GCODTG = p.CG / dt + TIMCON * p.CG * 2.0 - fx['dRng_dTg'] + fx['dHg_dTg'] + fx['dEg_dTg']
    GCORHS = fx['Rn_g0'] - TIMCON * p.CG * 2.0 * (Tg - Td) - fx['Hg0'] - fx['Eg0']

    DENOM = CCODTC * GCODTG - CCODTG * GCODTC
    DTc = (CCORHS * GCODTG - CCODTG * GCORHS) / DENOM
    DTg = (CCODTC * GCORHS - CCORHS * GCODTC) / DENOM

    Rn_c = fx['Rn_c0'] + fx['dRnc_dTc'] * DTc + fx['dRnc_dTg'] * DTg
    Rn_g = fx['Rn_g0'] + fx['dRng_dTc'] * DTc + fx['dRng_dTg'] * DTg
    Hc = fx['Hc0'] + fx['dHc_dTc'] * DTc + fx['dHc_dTg'] * DTg
    Hg = fx['Hg0'] + fx['dHg_dTc'] * DTc + fx['dHg_dTg'] * DTg
    Ec = fx['Ec0'] + fx['dEc_dTc'] * DTc + fx['dEc_dTg'] * DTg
    Eg = fx['Eg0'] + fx['dEg_dTc'] * DTc + fx['dEg_dTg'] * DTg
    G = TIMCON * p.CG * 2.0 * (Tg + DTg - Td)

    # Td: EXPLICITO, force-restore usando o Tg ja atualizado (igual ao SSiB)
    Td_novo = Td + dt * G / p.CD

    novo_estado = np.array([Tc + DTc, Tg + DTg, Td_novo])
    diag = dict(Rn_c=Rn_c, Rn_g=Rn_g, Hc=Hc, Hg=Hg, Ec=Ec, Eg=Eg, G=G)
    return novo_estado, diag


# ---------------------------------------------------------------------------
# RODADAS DE SIMULACAO E COMPARACAO
# ---------------------------------------------------------------------------
def rodar(metodo, nhoras=72, dt=3600.0, Tc0=295.0, Tg0=294.0, Td0=293.0, p=None):
    if p is None:
        p = Parametros()
    forc = gerar_forcante(nhoras=nhoras, dt=dt)
    n = len(forc['horas'])
    passo_fn = {'thomas': passo_tridiagonal,
                'matriz_cheia': passo_matriz_cheia,
                'ssib_2x2_explicito': passo_ssib_2x2_mais_explicito}[metodo]

    state = np.array([Tc0, Tg0, Td0])
    Tc_h = np.zeros(n); Tg_h = np.zeros(n); Td_h = np.zeros(n)
    for i in range(n):
        forc_i = (forc['Rg'][i], forc['Ta'][i], forc['qa'][i], forc['U'][i], forc['Ld'][i])
        state, diag = passo_fn(state, forc_i, p, dt)
        Tc_h[i], Tg_h[i], Td_h[i] = state

    return forc, dict(Tc=Tc_h, Tg=Tg_h, Td=Td_h)


if __name__ == "__main__":
    p = Parametros()

    forc, r_thomas = rodar('thomas', p=p)
    _, r_cheia = rodar('matriz_cheia', p=p)
    _, r_ssib = rodar('ssib_2x2_explicito', p=p)

    dif_tc = np.max(np.abs(r_thomas['Tc'] - r_cheia['Tc']))
    dif_tg = np.max(np.abs(r_thomas['Tg'] - r_cheia['Tg']))
    dif_td = np.max(np.abs(r_thomas['Td'] - r_cheia['Td']))
    print("=== Thomas (tridiagonal) x matriz cheia (devem ser ~identicos) ===")
    print(f"maior diferenca em Tc: {dif_tc:.2e} K")
    print(f"maior diferenca em Tg: {dif_tg:.2e} K")
    print(f"maior diferenca em Td: {dif_td:.2e} K")

    dif_td_ssib = np.max(np.abs(r_thomas['Td'] - r_ssib['Td']))
    print()
    print("=== Sistema 3x3 completo (Thomas) x atalho do SSiB (2x2 + Td explicito) ===")
    print(f"maior diferenca em Td ao longo das 72h: {dif_td_ssib:.4f} K")

    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 11})
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    h = forc['horas']

    axs[0].plot(h, r_thomas['Tc'] - 273.15, color="#27ae60", lw=2, label="Tc")
    axs[0].plot(h, r_thomas['Tg'] - 273.15, color="#e67e22", lw=2, label="Tg")
    axs[0].plot(h, r_thomas['Td'] - 273.15, color="#8e44ad", lw=2, label="Td")
    axs[0].set_ylabel("Temperatura (C)")
    axs[0].set_title("(A) Sistema tridiagonal 3x3 (Tc,Tg,Td) resolvido pelo algoritmo de Thomas")
    axs[0].legend(fontsize=9)

    axs[1].plot(h, (r_thomas['Tc'] - r_cheia['Tc']) * 1e6, color="#2980b9", label="dif. Tc (x1e6 K)")
    axs[1].plot(h, (r_thomas['Tg'] - r_cheia['Tg']) * 1e6, color="#c0392b", label="dif. Tg (x1e6 K)")
    axs[1].plot(h, (r_thomas['Td'] - r_cheia['Td']) * 1e6, color="#8e44ad", label="dif. Td (x1e6 K)")
    axs[1].set_ylabel("Diferenca (K x 1e-6)")
    axs[1].set_title("(B) Thomas x matriz cheia (numpy.linalg.solve) - validacao: diferenca ~ erro de arredondamento")
    axs[1].legend(fontsize=9)

    axs[2].plot(h, r_thomas['Td'] - 273.15, color="#8e44ad", lw=2, label="Td (sistema 3x3, Thomas)")
    axs[2].plot(h, r_ssib['Td'] - 273.15, color="#8e44ad", lw=1.2, ls="--",
                label="Td (atalho SSiB: 2x2 implicito + Td explicito)")
    axs[2].set_ylabel("Td (C)")
    axs[2].set_xlabel("Tempo (horas)")
    axs[2].set_title("(C) Solucao completa (3x3) x atalho operacional do SSiB (2x2 + explicito)")
    axs[2].legend(fontsize=9)

    for ax in axs:
        ax.set_xlim(0, h.max())
        ax.set_xticks(np.arange(0, h.max() + 1, 12))

    fig.suptitle("Tc, Tg, Td: sistema tridiagonal (Thomas) x matriz cheia x atalho do SSiB",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("tridiagonal_tc_tg_td.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print()
    print("Figura salva em tridiagonal_tc_tg_td.png")
