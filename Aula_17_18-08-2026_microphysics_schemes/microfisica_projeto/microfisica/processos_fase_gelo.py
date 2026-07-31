# -*- coding: utf-8 -*-
"""
processos_fase_gelo.py
========================

PASSO 2 do modelo de microfisica de nuvens: adiciona a categoria de
GELO DE NUVEM (cristais de gelo pequenos, subscrito 'i'): variaveis
prognosticas (qi, Ni) -- ja de 2 momentos no esquema OFICIAL de
Thompson (ao contrario de qc, qr, que so ganharam Nc, Nr como extensao
deste projeto -- ver `processos_chuva_quente.py`).

NOTACAO DO CURSO (slides MET-756-4, Parte IV -- equacoes de qi e Ni)
---------------------------------------------------------------------
    Pidsn  : nucleacao primaria de gelo (deposicao/condensacao-congelamento)
             -> ganha qi e Ni
    Pidep  : deposicao/sublimacao de vapor sobre cristais de gelo ja
             existentes -> ganha/perde qi (troca com qv)
    Pifzc  : congelamento heterogeneo (imersao) de goticulas de nuvem -> gelo
             -> perde qc,Nc; ganha qi,Ni
    Pimlt  : degelo do gelo de nuvem (T > 0 graus C) -> perde qi,Ni; ganha qc,Nc

ESCOPO DESTE PASSO (o que fica para o Passo 3)
-------------------------------------------------
Os slides tambem mostram os termos `Pispl` (multiplicacao de gelo por
rime splintering, Hallett-Mossop) e `Pi.iacw` (gelo colide e coleta
agua de nuvem, riming de cristais pequenos). Esses dois processos
dependem fisicamente de haver coleta de goticulas por particulas de
gelo em queda -- um efeito que so se torna importante quando graupel
(qg) existe (particulas maiores, caindo mais rapido, "varrendo" mais
goticulas). Por isso, `Pispl` e `Pi.iacw` sao adiados para o PASSO 3,
junto com `Picns` (autoconversao gelo->neve, que exige a categoria
qs) e `Picng` (conversao gelo->graupel). Isso e documentado aqui para
nao causar confusao ao comparar com a equacao completa de qi dos
slides.

O PROCESSO DE WEGENER-BERGERON-FINDEISEN (WBF)
-------------------------------------------------
Um dos processos mais importantes da fase mista NAO tem um simbolo P
proprio nas equacoes do curso -- ele e um EFEITO EMERGENTE da
combinacao entre `Pidep` e `Pccnd` (ver slides Parte III, secao
"Difusao: Processo de Wegener-Bergeron-Findeisen"): como a pressao de
saturacao do vapor sobre o GELO e sempre menor que sobre a AGUA
LIQUIDA (para T<0 graus C), um ambiente pode estar simultaneamente
SUBSATURADO em relacao a agua liquida e SUPERSATURADO em relacao ao
gelo. Nessa situacao, os cristais de gelo CRESCEM (Pidep>0) enquanto as
goticulas de nuvem EVAPORAM (Pccnd<0) para "alimentar" esse
crescimento -- mesmo sem nenhuma colisao entre gelo e agua.

Neste codigo, reproduzimos o WBF de forma simples e didatica through
a ORDEM dos processos dentro de cada passo de tempo (ver
`coluna_step2.py`): primeiro resolvemos Pidep usando a saturacao sobre
o GELO como referencia; DEPOIS resolvemos Pccnd usando a saturacao
sobre a AGUA LIQUIDA como referencia, mas ja com o vapor
"consumido" pelo gelo no passo anterior. Isso automaticamente produz
evaporacao de qc quando o ambiente estava perto da saturacao liquida e
o gelo "puxa" vapor para baixo do ponto de saturacao liquida -- exatamente
o efeito WBF. (Uma alternativa mais rigorosa, usada por MG2008 eq.
23-26, particiona Q explicitamente entre Pccnd e Pidep usando a razao
Fice=min(A/Q,1); optamos pela ordenacao sequencial aqui por
simplicidade didatica, e documentamos a diferenca.)
"""

import numpy as np
from .constantes import (Rv, cp, Lv, Ls, rho_i, rho_w, T0, MU_ICE,
                          QMIN, NMIN, gamma_func,
                          NIN0_COOPER, NIN_B_COOPER, T_NUCLEACAO_MAX,
                          TAU_ATIVACAO, D_GELO_NUCLEADO,
                          A_BIGG, B_BIGG, T_HOMOGENEO, Ka_AR, Dv_AR,
                          E_IC)
from .distribuicoes import lambda_gama, diametro_medio_numero


# =======================================================================
# PROPRIEDADES DE SATURACAO SOBRE O GELO
# =======================================================================
def pressao_saturacao_gelo(T):
    """
    Pressao de saturacao do vapor sobre GELO (Pa), formula de Magnus
    (Murphy & Koop 2005 / forma simplificada de Sonntag 1990), T em K.
    Note que esta e SEMPRE MENOR que a pressao de saturacao sobre agua
    liquida para T<0 graus C -- a base fisica do processo de
    Wegener-Bergeron-Findeisen (ver docstring do modulo).

        esi(T) = 611.15 * exp[22.452*(T-273.15) / (T-273.15+272.55)]   [Pa]
    """
    Tc = T - T0
    return 611.15 * np.exp(22.452 * Tc / (Tc + 272.55))


def razao_mistura_saturacao_gelo(T, p):
    """Razao de mistura de saturacao sobre gelo, qvi(T,p) [kg/kg]."""
    esi = pressao_saturacao_gelo(T)
    return 0.622 * esi / (p - esi)


# =======================================================================
# Pidsn -- NUCLEACAO PRIMARIA DE GELO (Cooper 1986)
# =======================================================================
def Pidsn(Ni, T, rho_ar, dt):
    """
    Pidsn -- nucleacao primaria de gelo por deposicao/condensacao-
    congelamento. Nas equacoes do curso:

        (dp*qi/dt) contem o termo +p* Pidsn      (ganha massa de gelo)
        (dp*Ni/dt) contem o termo +p* Pidsn/mi_nuc  (ganha numero de cristais)
        (dp*qv/dt) contem o termo -p* Pidsn      (perde vapor, implicito)

    Segue Cooper (1986), como usado em MG2008 eq. (20) e discutido nos
    slides da Parte III (modos de nucleacao de gelo, Vali 1999): a
    concentracao de nucleos de gelo ativos, NIN (em L^-1), e funcao so
    da temperatura:

        NIN(T) = 0.005 * exp[0.304*(T0 - T)]     [L^-1]

    limitada ao valor calculado em -35 graus C (209 L^-1, pois a extrapolacao
    de Cooper para temperaturas muito frias produz valores irrealistas).
    A nucleacao so ocorre para T < -5 graus C, e a concentracao de gelo atual
    (Ni) relaxa para NIN(T) numa escala de tempo de ativacao de 20 min
    (mesma escala usada para ativacao de CCN em MG2008).

    Retorna
    -------
    dqi_dt, dNi_dt : taxas (kg/kg/s, kg^-1 s^-1), >= 0 (so cria gelo).
    """
    if T >= T_NUCLEACAO_MAX:
        return 0.0, 0.0

    NIN_Lm1 = NIN0_COOPER * np.exp(NIN_B_COOPER * (T0 - T))
    NIN_max_Lm1 = NIN0_COOPER * np.exp(NIN_B_COOPER * 35.0)  # cap em -35 graus C (~209 L^-1)
    NIN_Lm1 = min(NIN_Lm1, NIN_max_Lm1)

    # converte NIN de L^-1 para kg^-1 (por kg de ar): 1 m^3 = 1000 L
    NIN_kg1 = NIN_Lm1 * 1000.0 / rho_ar

    deficit = NIN_kg1 - Ni
    if deficit <= 0:
        return 0.0, 0.0  # ja ha gelo suficiente, sem nucleacao adicional

    dNi_dt = deficit / TAU_ATIVACAO  # kg^-1 s^-1, relaxacao para NIN
    massa_cristal_novo = (np.pi / 6.0) * rho_i * D_GELO_NUCLEADO ** 3  # kg
    dqi_dt = dNi_dt * massa_cristal_novo  # kg/kg/s

    return dqi_dt, dNi_dt


# =======================================================================
# Pidep -- DEPOSICAO/SUBLIMACAO SOBRE CRISTAIS DE GELO EXISTENTES
# =======================================================================
def Pidep(qv, qi, Ni, T, p, rho_ar):
    """
    Pidep -- crescimento (deposicao, Pidep>0) ou encolhimento
    (sublimacao, Pidep<0) dos cristais de gelo ja existentes por
    difusao de vapor d'agua. Nas equacoes do curso:

        (dp*qv/dt) contem o termo -p* Pidep
        (dp*qi/dt) contem o termo +p* Pidep

    Equacao de crescimento por deposicao de uma particula esferica de
    gelo (slides Parte III, secao 6.3.2; Pruppacher & Klett 1997):

        dmi/dt = 4*pi*C*Swi / (Fk,i + Fd,i)

    onde C e a capacitancia (aproximada aqui por C=D/2, esfera efetiva
    de mesmo volume, "raio ri=rv" como discutido nos slides), Swi e a
    supersaturacao em relacao ao gelo, e:

        Fk,i = (Ls/(Rv*T)) * (Ls/(Ka*T) - 1)     (termo de calor/conducao)
        Fd,i = Rv*T / (Dv*esi(T))                 (termo de difusao de vapor)

    O diametro D usado e o diametro medio ponderado pelo numero da
    distribuicao gama assumida para o gelo (diagnosticado a partir de
    qi, Ni). A taxa total (bulk) e obtida multiplicando o crescimento
    de UM cristal (dmi/dt) pela concentracao numerica Ni.

    Retorna
    -------
    dqi_dt : taxa de variacao de qi (kg/kg/s, pode ser >0 ou <0)
    """
    if Ni <= NMIN:
        return 0.0

    esi = pressao_saturacao_gelo(T)
    qvi = razao_mistura_saturacao_gelo(T, p)
    Swi = (qv - qvi) / qvi if qvi > 0 else 0.0  # supersaturacao sobre o gelo

    if qi <= QMIN:
        # ainda nao ha massa mensuravel de gelo, mas Ni>0 (recem-nucleado):
        # usa o diametro assumido de nucleacao como estimativa inicial.
        D = D_GELO_NUCLEADO
    else:
        lam = lambda_gama(qi, Ni, rho_ar, rho_i, MU_ICE)
        D = diametro_medio_numero(lam, MU_ICE)

    C = D / 2.0  # capacitancia de uma esfera efetiva (aprox.)

    Fk = (Ls / (Rv * T)) * (Ls / (Ka_AR * T) - 1.0)
    Fd = (Rv * T) / (Dv_AR(T, p) * esi)

    dmi_dt = 4.0 * np.pi * C * Swi / (Fk + Fd)  # kg/s, por cristal

    dqi_dt = Ni * dmi_dt  # kg/kg/s (bulk)
    return dqi_dt


# =======================================================================
# Pifzc -- CONGELAMENTO HETEROGENEO (IMERSAO) DE GOTICULAS -> GELO
# =======================================================================
def Pifzc(qc, Nc, T, dt):
    """
    Pifzc -- congelamento heterogeneo de goticulas de nuvem (por
    imersao de um nucleo de congelamento dentro da goticula) formando
    novos cristais de gelo. Nas equacoes do curso:

        (dp*qc/dt) contem o termo -p* Pifzc
        (dp*qi/dt) contem o termo +p* Pifzc

    Para -40 graus C < T < 0 graus C: formula de Bigg (1953), na forma usada por
    Reisner et al. (1998, RRB) e discutida nos slides da Parte III
    ("Congelamento heterogeneo -- Congelamento por imersao"):

        taxa_por_goticula = B0 * exp[A0*(T0-T)] * Volume_goticula

    com A0=0.66 K^-1, B0=1e8 m^-3 s^-1 (equivalente as constantes
    classicas de Bigg, 100 cm^-3 s^-1). Cada goticula tem uma
    probabilidade de congelar proporcional ao seu volume e a uma funcao
    exponencial do resfriamento abaixo de 0 graus C.

    Para T <= -40 graus C: congelamento HOMOGENEO instantaneo (toda a agua de
    nuvem restante vira gelo neste mesmo passo de tempo) -- MG2008 e
    os slides da Parte III concordam que a nucleacao homogenea da fase
    liquida e o mecanismo dominante nessas temperaturas.

    Retorna
    -------
    dqc_dt, dNc_dt : taxas de perda de qc, Nc (kg/kg/s, kg^-1 s^-1; <=0)
    """
    if qc <= QMIN or Nc <= NMIN or T >= T0:
        return 0.0, 0.0

    if T <= T_HOMOGENEO:
        # congelamento homogeneo instantaneo: converte tudo neste passo
        dqc_dt = -qc / dt
        dNc_dt = -Nc / dt
        return dqc_dt, dNc_dt

    # congelamento heterogeneo (Bigg 1953)
    massa_goticula = qc / Nc  # kg por goticula
    volume_goticula = massa_goticula / rho_w  # m^3
    taxa_por_goticula = B_BIGG * np.exp(A_BIGG * (T0 - T)) * volume_goticula  # s^-1
    taxa_por_goticula = min(taxa_por_goticula, 1.0 / dt)  # nao mais que 100%/dt

    dNc_dt = -Nc * taxa_por_goticula
    dqc_dt = -qc * taxa_por_goticula  # mesma fracao de massa e numero congelam

    return dqc_dt, dNc_dt


# =======================================================================
# Pimlt -- DEGELO DO GELO DE NUVEM
# =======================================================================
def Pimlt(qi, Ni, T, dt):
    """
    Pimlt -- degelo (derretimento) do gelo de nuvem quando T > 0 graus C. Nas
    equacoes do curso:

        (dp*qi/dt) contem o termo -p* Pimlt
        (dp*qc/dt) contem o termo +p* Pimlt

    Para cristais de gelo pequenos (ao contrario de graupel/granizo,
    que tem um processo de degelo gradual descrito pela equacao de
    Rasmussen & Heymsfield 1987 nos slides da Parte III, Tabela 6.6.1),
    assume-se degelo INSTANTANEO: assim que T cruza 0 graus C, todo o gelo de
    nuvem vira agua de nuvem no mesmo passo de tempo (aproximacao
    padrao para cristais pequenos, cujo tempo de resposta termica e
    muito menor que o passo de tempo do modelo).

    Retorna
    -------
    dqi_dt, dNi_dt : taxas de perda de qi, Ni (kg/kg/s, kg^-1 s^-1; <=0)
    """
    if qi <= QMIN or T < T0:
        return 0.0, 0.0

    dqi_dt = -qi / dt
    dNi_dt = -Ni / dt
    return dqi_dt, dNi_dt


# =======================================================================
# VELOCIDADE TERMINAL DO GELO DE NUVEM (para sedimentacao e coleta)
# =======================================================================
def velocidade_terminal_gelo(qi, Ni, rho_ar):
    """
    Velocidade terminal de queda do gelo de nuvem (m/s) -- MUITO menor
    que a da chuva, pois cristais de gelo pequenos caem lentamente
    (tipicamente poucos cm/s, ver p.ex. Heymsfield 1972; Ono 1969).

    Relacao de potencia simplificada V(D) = a_i * D^b_i, com
    a_i=700 s^-1*m^(1-b_i), b_i=1 (aproximacao linear simples,
    suficiente para fins didaticos -- esquemas operacionais usam
    relacoes massa-diametro e area-diametro mais elaboradas, ver
    Thompson et al. 2008).

    Retorna
    -------
    Vq, Vn : velocidades terminais ponderadas por massa e por numero (m/s)
    """
    if qi <= QMIN or Ni <= NMIN:
        return 0.0, 0.0

    a_i, b_i = 700.0, 1.0
    mu = MU_ICE
    lam = lambda_gama(qi, Ni, rho_ar, rho_i, mu)

    Vq = a_i * (gamma_func(mu + 4.0 + b_i) / gamma_func(mu + 4.0)) * lam ** (-b_i)
    Vn = a_i * (gamma_func(mu + 1.0 + b_i) / gamma_func(mu + 1.0)) * lam ** (-b_i)

    Vq = min(Vq, 1.5)  # limite fisico superior para cristais pequenos
    Vn = min(Vn, 1.5)
    return Vq, Vn


# =======================================================================
# Pi_iacw -- GELO COLETA AGUA DE NUVEM (riming de cristais pequenos)
# =======================================================================
def Pi_iacw(qc, Ni, qi, T, rho_ar):
    """
    Pi_iacw -- "ICE Accretes Cloud Water": cristais de gelo em queda
    coletam goticulas de nuvem super-resfriadas em sua trajetoria
    (riming). Nas equacoes do curso:

        (dp*qc/dt) contem o termo -p* Pi_iacw
        (dp*qi/dt) contem o termo +p* Pi_iacw

    *** ADIADO DO PASSO 2 para o PASSO 3 *** (ver docstring do modulo
    `processos_fase_gelo.py`, secao "Escopo deste passo"): implementado
    aqui porque agora existe uma velocidade terminal de gelo definida
    (`velocidade_terminal_gelo`) para calcular a taxa de varredura.

    Usa a forma de colecao continua simplificada (ver
    `processos_fase_mista.colecao_continua`): a goticula de nuvem e
    assumida parada (V_alvo=0) frente a velocidade de queda do gelo.

    Retorna
    -------
    dqc_dt : taxa de perda de qc (kg/kg/s, <=0)
    """
    if qc <= QMIN or Ni <= NMIN:
        return 0.0

    if qi <= QMIN:
        D = D_GELO_NUCLEADO
    else:
        lam = lambda_gama(qi, Ni, rho_ar, rho_i, MU_ICE)
        D = diametro_medio_numero(lam, MU_ICE)

    Vi, _ = velocidade_terminal_gelo(qi, Ni, rho_ar)

    dqc_dt = -E_IC * qc * Ni * (np.pi / 4.0) * D ** 2 * abs(Vi - 0.0)
    return dqc_dt
