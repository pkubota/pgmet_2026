# -*- coding: utf-8 -*-
"""
processos_chuva_quente.py
==========================

PASSO 1 do modelo de microfisica de nuvens: esquema de "chuva quente"
(warm rain), com duas categorias de agua condensada:

    - Agua de nuvem  (goticulas, subscrito 'c'): variaveis (qc, Nc)
    - Agua de chuva  (gotas de chuva, subscrito 'r'): variaveis (qr, Nr)

NOTACAO DO CURSO (slides MET-756-4, "Parametrizacao de Microfisica de
Nuvens de Gregory Thompson", baseada em Reisner et al. 1998 -- RRB)
--------------------------------------------------------------------
As equacoes governantes do esquema de Thompson, mostradas nos slides,
escrevem os termos-fonte/sumidouro de cada categoria usando o simbolo
P (de "Processo"), com um subscrito de duas ou tres letras que
identifica o par de categorias envolvidas. Para a fase quente
(liquida), os termos relevantes que aparecem nas equacoes de qv, qc e
qr do curso sao:

    Pccnd : condensacao/evaporacao de agua de nuvem (ajuste de saturacao)
            -> aparece com sinal (+) na equacao de qc e (-) na de qv
    Pccnr : autoconversao de agua de nuvem em chuva (cloud -> rain)
            -> aparece com sinal (-) em qc e (+) em qr
    Pracw : acrescimo -- chuva "accreta" (coleta) agua de nuvem (rain
            ACcretes Cloud Water)
            -> aparece com sinal (-) em qc e (+) em qr
    Prevp : evaporacao da chuva (Rain EVaPoration)
            -> aparece com sinal (+) em qv e (-) em qr
    Prprc : fluxo de sedimentacao/precipitacao de qr (termo de divergencia
            vertical do fluxo de queda, escrito SEM o fator p* nas
            equacoes do curso, pois e um termo de transporte vertical
            e nao uma "fonte" microfisica local)

Este modulo implementa cada um desses termos como uma funcao Python
com o MESMO NOME usado nos slides (Pccnd, Pccnr, Pracw, Prevp), para
que a correspondencia entre codigo e a formulacao vista em aula seja
direta. As formulas numericas usadas dentro de cada funcao seguem a
literatura consolidada (Khairoutdinov & Kogan 2000 para Pccnr e Pracw,
como usado tambem em Morrison & Gettelman 2008 eq. 27-28), ja que os
slides do curso apresentam a estrutura das equacoes de prognostico mas
nao fixam uma formula numerica unica para cada P (isso e uma escolha
do esquema/implementacao, como o proprio Thompson et al. 2004/2008
fazem com suas proprias formulas).

EXTENSAO DE DOIS MOMENTOS (alem do esquema "oficial" de Thompson)
------------------------------------------------------------------
O esquema de Thompson descrito nos slides e de UM MOMENTO para nuvem e
chuva (so qc, qr -- nao ha Nc nem Nr nas equacoes governantes: apenas
Ni, Ns, Ng sao prognosticos). Neste projeto optamos por MANTER Nc e Nr
como variaveis prognosticas extras (2 momentos tambem para a fase
quente), pois isso permite calcular diametros efetivos e velocidades
terminais mais realistas e ilustrar conceitos importantes da disciplina
(ex.: autocolecao de chuva, largura do espectro). A funcao `Pr_self`
(autocolecao da chuva) e as funcoes de `distribuicoes.py` para
diagnosticar o espectro NAO tem um simbolo P correspondente nas
equacoes oficiais do curso porque o esquema de 1 momento nao precisa
delas; sao um "plus" do nosso modelo, e isso esta documentado
explicitamente para nao causar confusao na hora de comparar com os
slides.

PROCESSOS IMPLEMENTADOS NESTE ARQUIVO:

    1. Pccnd    -> condensacao/evaporacao de qc (ajuste de saturacao)
    2. Pccnr    -> autoconversao (qc -> qr)
    3. Pracw    -> acrescimo (chuva coleta agua de nuvem)
    4. Pr_self  -> autocolecao da chuva [EXTENSAO 2 momentos, sem P oficial]
    5. Prevp    -> evaporacao da chuva
    6. velocidade_terminal_chuva -> V(D) para o termo de sedimentacao Prprc

Convencao de sinais: todas as funcoes abaixo retornam a taxa
(kg/kg/s, ou kg^-1 s^-1 para numero) JA COM O SINAL de perda/ganho para
a variavel no nome da funcao (ex.: Pccnr retorna um valor <= 0, pois
representa perda de qc; o driver principal em coluna_step1.py aplica o
sinal oposto (+) para a variavel que recebe a massa, exatamente como
nas equacoes do curso, onde o mesmo simbolo Pccnr aparece com sinal (-)
em qc e (+) em qr).
"""

import numpy as np
from .constantes import (Rd, Rv, cp, Lv, rho_w, T0, MU_CLOUD, MU_RAIN,
                          QMIN, NMIN, gamma_func)
from .distribuicoes import lambda_gama, N0_gama


# =======================================================================
# 1) Pccnd -- CONDENSACAO / EVAPORACAO DE AGUA DE NUVEM
# =======================================================================
def pressao_saturacao_agua(T):
    """
    Pressao de saturacao do vapor sobre agua liquida (Pa), formula de
    Clausius-Clapeyron / Bolton (1980), T em Kelvin.

        es(T) = 611.2 * exp[17.67*(T-273.15) / (T-29.65)]   [Pa]
    """
    Tc = T - T0
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def razao_mistura_saturacao(T, p):
    """
    Razao de mistura de saturacao qvs(T,p) [kg/kg], a partir da pressao
    de saturacao es(T) e da pressao total p (Pa):

        qvs = 0.622 * es / (p - es)
    """
    es = pressao_saturacao_agua(T)
    return 0.622 * es / (p - es)


def dqvs_dT(T, p, h=0.01):
    """Derivada numerica de qvs em relacao a T (para o ajuste de saturacao)."""
    return (razao_mistura_saturacao(T + h, p) - razao_mistura_saturacao(T - h, p)) / (2 * h)


def Pccnd(qv, qc, T, p, dt):
    """
    Pccnd -- condensacao/evaporacao de agua de nuvem (ajuste de
    saturacao). Nas equacoes do curso:

        (dp*qv/dt) contem o termo -p* Pccnd        (perde vapor quando Pccnd>0)
        (dp*qc/dt) contem o termo +p* Pccnd        (ganha agua de nuvem quando Pccnd>0)

    Implementacao (ajuste de saturacao simplificado, analogo ao termo Q
    de Morrison & Gettelman 2008 eq. 23-26, sem separacao por fracao de
    nuvem Fcld -- aqui assumimos Fcld=1 dentro/fora da nuvem):

    Se o ar esta supersaturado (qv > qvs), o excesso de vapor condensa
    (aproximacao de ajuste de saturacao instantaneo, padrao quando nao
    se resolve explicitamente o crescimento difusivo gota a gota).
    Se o ar esta subsaturado (qv < qvs) e existe qc>0, evapora-se agua
    de nuvem ate a saturacao ou ate qc=0 (o que ocorrer primeiro).

    O calor latente liberado/consumido aquece/esfria o ar (dT).

    Retorna
    -------
    dqc, dqv, dT : variacoes JA INTEGRADAS no passo de tempo dt (kg/kg,
                   kg/kg, K) -- nao sao taxas por segundo.
                   dqc > 0 equivale a Pccnd > 0 (condensacao);
                   dqc < 0 equivale a Pccnd < 0 (evaporacao de nuvem).
    """
    qvs = razao_mistura_saturacao(T, p)
    Q = (qv - qvs)  # supersaturacao/subsaturacao disponivel (kg/kg)

    if Q > 0:
        # Condensacao: uma fracao de Q vira agua liquida (correcao
        # psicrometrica pelo calor latente liberado, 1 iteracao linear).
        dqc = Q / (1.0 + (Lv / cp) * dqvs_dT(T, p))
    else:
        # Evaporacao: no maximo evapora-se |Q|, limitado pela qc disponivel
        dqc = max(Q, -qc)  # dqc <= 0

    dqv = -dqc
    dT = (Lv / cp) * dqc  # libera calor latente se dqc>0 (condensacao)

    return dqc, dqv, dT


# =======================================================================
# 2) Pccnr -- AUTOCONVERSAO: NUVEM -> CHUVA (Khairoutdinov & Kogan 2000)
# =======================================================================
def Pccnr(qc, Nc, rho_ar):
    """
    Pccnr -- autoconversao de agua de nuvem em chuva (goticulas colidem
    entre si e crescem ate formar gotas de chuva "embrionarias"). Nas
    equacoes do curso:

        (dp*qc/dt) contem o termo -p* Pccnr       (perde agua de nuvem)
        (dp*qr/dt) contem o termo +p* Pccnr       (ganha agua de chuva)

    Formula numerica: Khairoutdinov & Kogan (2000), tambem usada por
    Morrison & Gettelman (2008) eq. (27) (aqui sem o fator de correcao
    de variabilidade subgrid -- adequado para uma coluna/parcela sem
    variabilidade subgrid, como no Passo 1):

        Pccnr = 1350 * qc^2.47 * Nc^(-1.79)     [g/g/s], qc em g/g e
                                                  Nc em cm^-3

    IMPORTANTE (unidades): a formula empirica do KK2000 EXIGE que qc
    esteja em g/g e Nc em cm^-3. A conversao e feita aqui dentro,
    devolvendo o resultado em kg/kg/s (unidades SI) para o resto do
    codigo.

    Retorna
    -------
    dqc_dt : taxa de perda de qc (kg/kg/s, <=0)  -- equivale a -Pccnr
    dNc_dt : taxa de perda de Nc (kg^-1 s^-1, <=0), pela conversao de
             massa em gotas "embrionarias" de diametro D_auto ~ 25 um.
    """
    if qc <= QMIN or Nc <= NMIN:
        return 0.0, 0.0

    # --- conversao de unidades para a formula empirica ---
    qc_gg = qc  # kg/kg e numericamente igual a g/g (razao adimensional)
    Nc_cm3 = Nc * rho_ar * 1.0e-6  # [kg^-1]*[kg/m^3]*[m^3/cm^3->1e-6] = cm^-3

    dqc_dt = -1350.0 * qc_gg ** 2.47 * Nc_cm3 ** (-1.79)  # g/g/s == kg/kg/s

    # Concentracao numerica perdida por qc: assume-se um diametro medio
    # de "gota embrionaria" formada (D_auto ~ 25 micra, valor tipico
    # usado em varios esquemas, e.g. Seifert & Beheng 2001) para
    # converter massa perdida em numero de gotas de chuva formadas.
    D_auto = 25.0e-6  # m
    massa_gota = (np.pi / 6.0) * rho_w * D_auto ** 3  # kg por gota
    dNr_dt_formadas = -dqc_dt / massa_gota  # > 0, numero de gotas formadas/kg ar/s

    dNc_dt = -dNr_dt_formadas  # Nc perde o numero equivalente de embrioes

    return dqc_dt, dNc_dt


# =======================================================================
# 3) Pracw -- ACRESCIMO: CHUVA COLETA GOTICULAS DE NUVEM
# =======================================================================
def Pracw(qc, qr):
    """
    Pracw -- "Rain ACcretes Cloud Water": gotas de chuva ja existentes
    "varrem" o caminho e coletam goticulas de nuvem menores. Nas
    equacoes do curso:

        (dp*qc/dt) contem o termo -p* Pracw       (perde agua de nuvem)
        (dp*qr/dt) contem o termo +p* Pracw       (ganha agua de chuva)

    Formula numerica: Khairoutdinov & Kogan (2000), usada em MG2008
    eq. (28):

        Pracw = 67 * (qc * qr)^1.15    [g/g/s], qc,qr em g/g

    Como qc e qr entram com o MESMO expoente, o resultado e
    dimensionalmente consistente em kg/kg/s sem conversao adicional de
    unidades (diferente de Pccnr, que tem expoentes distintos).

    A concentracao numerica de goticulas de nuvem (Nc) diminui na MESMA
    proporcao que qc durante este processo (mesmo pressuposto de
    MG2008: o tamanho medio da distribuicao de nuvem nao muda).

    Retorna
    -------
    dqc_dt : taxa de perda de agua de nuvem por acrescimo (kg/kg/s, <=0)
             -- equivale a -Pracw
    """
    if qc <= QMIN or qr <= QMIN:
        return 0.0

    dqc_dt = -67.0 * (qc * qr) ** 1.15
    return dqc_dt


# =======================================================================
# 4) Pr_self -- AUTOCOLECAO DA CHUVA (EXTENSAO de 2 momentos, sem P oficial)
# =======================================================================
def Pr_self(qr, Nr, rho_ar):
    """
    Pr_self -- autocolecao (self-collection) de gotas de chuva: duas
    gotas de chuva colidem e se fundem em uma gota maior.

    *** ESTE TERMO NAO EXISTE nas equacoes oficiais do esquema de
    Thompson mostradas nos slides (que e de 1 momento para qr, sem Nr
    prognostico). Ele e uma EXTENSAO deste projeto para permitir o
    tratamento de 2 momentos tambem na chuva. ***

    Nao altera a massa total de chuva (qr fica igual), mas REDUZ o
    numero de gotas (Nr diminui), pois duas viram uma so.
    Parametrizacao de Beheng (1994), citada tambem em MG2008 secao "g":

        (dNr/dt)_self = -k_rr * Nr * qr * rho_ar

    Retorna
    -------
    dNr_dt : taxa de variacao de Nr (kg^-1 s^-1, <=0)
    """
    if qr <= QMIN or Nr <= NMIN:
        return 0.0

    k_rr = 5.78  # constante empirica de Beheng (1994)
    dNr_dt = -k_rr * Nr * qr * rho_ar
    return dNr_dt


# =======================================================================
# 5) Prevp -- EVAPORACAO DA CHUVA (abaixo da base da nuvem, ar subsaturado)
# =======================================================================
def Prevp(qr, Nr, qv, T, p, rho_ar):
    """
    Prevp -- "Rain EVaPoration": evaporacao de gotas de chuva caindo
    atraves de ar subsaturado (processo-chave para "virga": chuva que
    evapora antes de atingir o solo). Nas equacoes do curso:

        (dp*qv/dt) contem o termo +p* Prevp        (ganha vapor)
        (dp*qr/dt) contem o termo -p* Prevp        (perde agua de chuva)

    Forma simplificada e didatica (o tratamento completo de ventilacao
    de Rogers & Yau / Pruppacher & Klett, analogo ao usado por Thompson
    et al. 2004, integra sobre a distribuicao gama de tamanhos e inclui
    o efeito de ventilacao; aqui aproximamos por uma dependencia de
    potencia na subsaturacao e na area superficial das gotas):

        Prevp = C_evap * (qvs - qv) * qr^(2/3)

    Retorna
    -------
    dqr_dt : taxa (kg/kg/s, <=0). NAO limitado internamente por -qr/dt
             -- ver nota de estabilidade abaixo.
    dNr_dt : taxa de Nr (kg^-1 s^-1, <=0), proporcional a razao Nr/qr
             (gotas encolhem e desaparecem na mesma proporcao de massa).

    NOTA DE ESTABILIDADE NUMERICA (licao aprendida durante o
    desenvolvimento deste projeto)
    ------------------------------------------------------------------
    Esta funcao retorna uma TAXA (kg/kg/s). Nao aplicamos aqui nenhum
    limite do tipo "max(dqr_dt, -qr)", pois isso compararia uma TAXA
    (unidade /s) com uma QUANTIDADE (qr, sem unidade de tempo) -- uma
    comparacao dimensionalmente inconsistente. Se essa taxa "limitada"
    for multiplicada por dt no chamador, o resultado pode remover mais
    massa de chuva do que realmente existe quando dt > 1s. O limite
    fisico correto ("nao evaporar mais chuva do que existe") DEVE ser
    aplicado no chamador (`coluna_step1.py`), DEPOIS de multiplicar a
    taxa por dt, usando o MESMO valor ja limitado tanto para reduzir qr
    quanto para aumentar qv e ajustar T. Esse exato bug (limitar apenas
    qr, mas usar o valor bruto/nao-limitado para atualizar qv) foi
    encontrado durante a validacao deste modelo com o teste de
    conservacao de massa (`teste_conservacao.py`): ele criava massa de
    agua "do nada", produzindo mais de 2000 mm de chuva irreais em 1h
    de simulacao. Fica documentado aqui como licao pratica de
    verificacao numerica em esquemas de microfisica.
    """
    if qr <= QMIN:
        return 0.0, 0.0

    qvs = razao_mistura_saturacao(T, p)
    subsat = qvs - qv
    if subsat <= 0:
        return 0.0, 0.0  # ar saturado/supersaturado: chuva nao evapora

    C_evap = 4.0e3  # coeficiente empirico simplificado (ajustavel / didatico)
    dqr_dt = -C_evap * subsat * (qr ** (2.0 / 3.0))

    if qr > QMIN:
        dNr_dt = dqr_dt * (Nr / qr)
    else:
        dNr_dt = 0.0

    return dqr_dt, dNr_dt


# =======================================================================
# 6) VELOCIDADE TERMINAL DE QUEDA (usada no termo de sedimentacao Prprc)
# =======================================================================
def velocidade_terminal_chuva(qr, Nr, rho_ar, rho_ar_ref=1.2):
    """
    Velocidade terminal de queda media das gotas de chuva, usada para
    calcular o termo de sedimentacao (chamado "Prprc" -- e "Nrprc" para
    numero, seguindo o mesmo padrao de nomenclatura usado nos slides
    para Nsprc e Ngprc, os fluxos de sedimentacao de numero de neve e
    graupel).

    Relacao de potencia D-V classica (Lin et al. 1983):

        V(D) = a * D^b * (rho_ar_ref/rho_ar)^0.5

    com a=842 m^(1-b)/s, b=0.8 (valores tipicos para chuva).

    A velocidade media ponderada pela massa (Vq, usada para o fluxo de
    qr = Prprc) e pelo numero (Vn, usada para o fluxo de Nr = "Nrprc",
    extensao de 2 momentos) sao obtidas integrando V(D)*D^k*N(D) sobre
    a distribuicao gama assumida.

    Retorna
    -------
    Vq : velocidade terminal ponderada pela massa (m/s) -> usada em Prprc
    Vn : velocidade terminal ponderada pelo numero (m/s) -> usada em Nrprc
    """
    if qr <= QMIN or Nr <= NMIN:
        return 0.0, 0.0

    a, b = 842.0, 0.8
    mu = MU_RAIN
    lam = lambda_gama(qr, Nr, rho_ar, rho_w, mu)

    correcao_rho = (rho_ar_ref / rho_ar) ** 0.5

    Vq = a * (gamma_func(mu + 4.0 + b) / gamma_func(mu + 4.0)) * lam ** (-b) * correcao_rho
    Vn = a * (gamma_func(mu + 1.0 + b) / gamma_func(mu + 1.0)) * lam ** (-b) * correcao_rho

    # limite fisico superior (gotas grandes nao excedem ~9-10 m/s)
    Vq = min(Vq, 9.5)
    Vn = min(Vn, 9.5)
    return Vq, Vn
