# -*- coding: utf-8 -*-
"""
MET-576-4 - Fontes de Erros nos Modelos de Previsao Numerica
Trabalho: Estabilidade e dissipacao em esquemas de integracao temporal
para modelos oceanicos: Leapfrog, Adams-Bashforth e Runge-Kutta

Grupo: Jose Augusto Ferreira Neto, Natacha Pires Ramos,
       Guilherme Lopes Mermejo, Pietra Santos de Sa Lisboa,
       Lucas Costa Antunes

Modelo idealizado: coluna d'agua oceanica, equacoes de Boussinesq
linearizadas reduzidas a um unico modo vertical (oscilador de empuxo):

    dw/dt =  b
    db/dt = -N^2 w

onde w e a perturbacao de velocidade vertical, b a perturbacao de
empuxo (buoyancy) e N a frequencia de Brunt-Vaisala (constante:
estratificacao idealizada uniforme). Este e o mecanismo restaurador
das ondas internas de gravidade em sua forma mais simples (um unico
modo vertical, sem propagacao horizontal).

Solucao analitica exata (dado w(0)=w0, b(0)=b0):
    w(t) = w0*cos(N t) + (b0/N)*sin(N t)
    b(t) = b0*cos(N t) - N*w0*sin(N t)

Energia mecanica por unidade de massa, EXATAMENTE conservada no
continuo:
    E(t) = 1/2 * [ w(t)^2 + b(t)^2/N^2 ] = E(0)  para todo t

    KE  = 1/2 w^2           (energia cinetica)
    APE = 1/2 b^2 / N^2     (energia potencial disponivel)

Reducao complexa (mesma tecnica usada no material do G2 do MET-579,
adaptada aqui para o oscilador de empuxo em vez da oscilacao inercial):
definindo z = N w + i b, tem-se
    dz/dt = N dw/dt + i db/dt = N b + i(-N^2 w) = -i N (N w + i b) = -i N z
ou seja, o mesmo tipo de EDO linear complexa dz/dt = -i N z que aparece
na oscilacao inercial (la com z = u + i v e frequencia f). Isso permite
reaproveitar toda a maquinaria de analise de estabilidade linear
(fator de amplificacao lambda(x), x = N dt) ja desenvolvida para o G2.
"""

import numpy as np


def solucao_analitica(t, w0, b0, N):
    """Solucao exata do oscilador de empuxo no instante t."""
    w = w0 * np.cos(N * t) + (b0 / N) * np.sin(N * t)
    b = b0 * np.cos(N * t) - N * w0 * np.sin(N * t)
    return w, b


def energia(w, b, N):
    """Energia mecanica total, cinetica e potencial disponivel (por unidade de massa)."""
    ke = 0.5 * w ** 2
    ape = 0.5 * (b ** 2) / (N ** 2)
    return ke + ape, ke, ape


def derivada(w, b, N):
    """dw/dt, db/dt do oscilador de empuxo (lado direito do sistema)."""
    return b, -(N ** 2) * w


# ---------------------------------------------------------------------
# Esquema de um unico passo: Runge-Kutta classico de 4a ordem
# ---------------------------------------------------------------------

def rk4(w, b, N, dt):
    """
    RK4 classico. Esquema de um unico passo (self-starting), SEM modo
    computacional -- motivo pelo qual nao pode ser usado para demonstrar
    o efeito do filtro de Robert-Asselin (essa demonstracao exige um
    esquema de multiplos niveis de tempo, como o Leapfrog).
    """
    k1_w, k1_b = derivada(w, b, N)
    k2_w, k2_b = derivada(w + 0.5 * dt * k1_w, b + 0.5 * dt * k1_b, N)
    k3_w, k3_b = derivada(w + 0.5 * dt * k2_w, b + 0.5 * dt * k2_b, N)
    k4_w, k4_b = derivada(w + dt * k3_w, b + dt * k3_b, N)

    w_novo = w + (dt / 6.0) * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)
    b_novo = b + (dt / 6.0) * (k1_b + 2 * k2_b + 2 * k3_b + k4_b)
    return w_novo, b_novo


# ---------------------------------------------------------------------
# Esquema de multiplos passos explicito: Adams-Bashforth de 2a ordem
# ---------------------------------------------------------------------

def adams_bashforth2(w_atu, b_atu, dw_ant, db_ant, N, dt):
    """
    Adams-Bashforth de 2a ordem (AB2):
        y_(n+1) = y_n + dt/2 * (3 f_n - f_(n-1))

    Precisa da derivada do passo ANTERIOR (dw_ant, db_ant = f_(n-1)),
    logo nao e self-starting: o primeiro passo precisa ser bootstrapped
    (ver 'inicializar_ab2' abaixo).

    Ao contrario do Leapfrog, o AB2 tambem tem uma raiz espuria na sua
    equacao caracteristica, mas essa raiz tende a ZERO quando dt -> 0
    (e permanece pequena, tipicamente << 1, para dt dentro da faixa de
    estabilidade) -- portanto o "modo computacional" do AB2 e fortemente
    amortecido a cada passo, ao contrario do modo computacional do
    Leapfrog, que e neutro (|lambda_comp| = 1) e por isso persiste
    indefinidamente sem um filtro. Essa e a distincao central pedida no
    entregavel diferencial deste trabalho.

    Retorna (w_novo, b_novo, dw_n, db_n) -- os dois ultimos sao a
    derivada calculada em n, que devera ser passada como dw_ant/db_ant
    na proxima chamada.
    """
    dw_atu, db_atu = derivada(w_atu, b_atu, N)
    w_novo = w_atu + (dt / 2.0) * (3.0 * dw_atu - dw_ant)
    b_novo = b_atu + (dt / 2.0) * (3.0 * db_atu - db_ant)
    return w_novo, b_novo, dw_atu, db_atu


def inicializar_ab2(w0, b0, N, dt, modo="rk4"):
    """
    Calcula a derivada f_(-1) = (dw_ant, db_ant) necessaria para dar a
    largada do AB2 no primeiro passo real (n=0 -> n=1), segundo dois
    modos possiveis:

    modo='rk4':
        Um unico passo de RK4 (de altissima precisao, O(dt^4)) e usado
        para gerar o estado em t=-dt a partir do estado em t=0
        integrado "para tras" no tempo (dt -> -dt); a derivada nesse
        estado retroativo e entao usada como f_(-1). Erro de
        inicializacao desprezivel frente ao erro de truncamento O(dt^2)
        do proprio AB2 -- e a escolha default e recomendada na pratica.

    modo='euler':
        Um unico passo de Euler explicito (O(dt), menos preciso) para
        tras no tempo. Serve para o exercicio de comparar o impacto da
        qualidade da inicializacao no erro global do AB2 (ao contrario
        do Leapfrog, cujo erro de inicializacao persiste sem
        amortecimento, o AB2 tem raiz espuria amortecida, entao o
        exercicio pede para o aluno verificar se esse erro de fato
        desaparece mais rapido aqui).

    Retorna (dw_m1, db_m1): a derivada no instante t = -dt.
    """
    if modo == "rk4":
        w_m1, b_m1 = rk4(w0, b0, N, -dt)
    elif modo == "euler":
        dw0, db0 = derivada(w0, b0, N)
        w_m1 = w0 - dt * dw0
        b_m1 = b0 - dt * db0
    else:
        raise ValueError("modo deve ser 'rk4' ou 'euler'")

    dw_m1, db_m1 = derivada(w_m1, b_m1, N)
    return dw_m1, db_m1


# ---------------------------------------------------------------------
# Esquema de multiplos niveis de tempo: Leapfrog (CTCS)
# ---------------------------------------------------------------------

def leapfrog(w_ant, b_ant, w_atu, b_atu, N, dt):
    """
    Leapfrog (Centered-in-Time):
        w_(n+1) = w_(n-1) + 2 dt b_n
        b_(n+1) = b_(n-1) - 2 dt N^2 w_n

    Precisa de DOIS niveis de tempo anteriores (n-1 e n): nao
    self-starting (ver 'inicializar_leapfrog'). Neutro em amplitude no
    modo fisico (|lambda_fis| = 1 exatamente, estavel para N dt < 1),
    mas introduz um MODO COMPUTACIONAL espurio com |lambda_comp| = 1
    tambem -- o classico ruido "2 dt" que oscila de sinal a cada passo
    e NAO se amortece sozinho, ao contrario do modo espurio do AB2.
    Essa e a razao para aplicar o filtro de Robert-Asselin.
    """
    w_novo = w_ant + 2.0 * dt * b_atu
    b_novo = b_ant - 2.0 * dt * (N ** 2) * w_atu
    return w_novo, b_novo


def robert_asselin_filtro(y_atu, y_novo, y_atu_filtrado_ant, alpha=0.1):
    """
    Filtro de Robert-Asselin classico (Asselin, 1972), aplicado ao
    nivel de tempo n APOS o Leapfrog ter avancado para n+1:

        y_n_filtrado = y_n + alpha * (y_(n-1)_filtrado - 2 y_n + y_(n+1))

    alpha tipico operacional entre 0.01 e 0.05; valores maiores
    suprimem o modo computacional mais rapido, mas amortecem tambem o
    modo fisico (ver oceano_coluna_modo_computacional.py para o
    diagnostico quantitativo desse compromisso).
    """
    return y_atu + alpha * (y_atu_filtrado_ant - 2.0 * y_atu + y_novo)


def inicializar_leapfrog(w0, b0, N, dt, modo="rk4"):
    """
    Calcula o par (w_-1, b_-1) necessario para dar a largada do
    Leapfrog, segundo dois modos de inicializacao:

    modo='rk4':   um passo de RK4 (para tras no tempo) -- inicializacao
                  de alta precisao, recomendada na pratica.
    modo='euler': um passo de Euler explicito (para tras) -- erro de
                  O(dt) que, ao contrario do AB2, NAO e amortecido pelo
                  Leapfrog (modo fisico neutro), permanecendo como um
                  vies visivel por toda a integracao.
    modo='analitico': usa a propria solucao analitica em t=-dt
                  (inicializacao "perfeita", possivel apenas porque
                  conhecemos a solucao exata deste problema-teste).

    Retorna (w_m1, b_m1).
    """
    if modo == "rk4":
        return rk4(w0, b0, N, -dt)
    elif modo == "euler":
        dw0, db0 = derivada(w0, b0, N)
        return w0 - dt * dw0, b0 - dt * db0
    elif modo == "analitico":
        return solucao_analitica(-dt, w0, b0, N)
    else:
        raise ValueError("modo deve ser 'rk4', 'euler' ou 'analitico'")
