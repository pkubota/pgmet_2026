# -*- coding: utf-8 -*-
"""
=====================================================================
 MODELO DIDATICO DE SUPERFICIE VEGETADA
 MET-576-4 - Modelagem Numerica da Atmosfera - Dr. Paulo Yoshio Kubota
 Baseado no material "Modelo com vegetacao" (vegetacao.pdf):
   - Balanco hidrico (interceptacao + solo)
   - Interceptacao de agua na folha (M, S, W=M/S)
   - Transpiracao (resistencia estomatica rs, resistencia de copa rc)
   - Balanco de radiacao (fracao vegetada V, albedo e emissividade)
   - Fluxo de momentum (perfil logaritmico, ra)
   - Sistema solo-planta-atmosfera (balanco de energia da superficie)
=====================================================================

Este script segue a MESMA notacao das equacoes apresentadas em aula,
para que cada linha do codigo possa ser associada diretamente a um
slide. E uma versao SIMPLIFICADA (modelo de "grande folha" / big-leaf
com temperatura de pele unica T), adequada para demonstracao em sala.

---------------------------------------------------------------------
 DOIS METODOS NUMERICOS - EXPLICITO x IMPLICITO (destaque deste script)
---------------------------------------------------------------------
A equacao prognostica da temperatura de superficie,

    C dT/dt = Rn(T) - G(T) - H(T) - L[Ei(T)+Ec(T)+Es(T)]

e uma EDO no tempo em que TODOS os termos do lado direito dependem da
propria temperatura T que se quer calcular. Existem duas formas
classicas de discretiza-la; este script implementa as DUAS, lado a
lado, para fins de comparacao:

  (A) METODO EXPLICITO (Euler progressivo / "forward Euler")
      -> funcao passo_explicito()
      Os fluxos do lado direito sao avaliados com a temperatura DO
      INICIO do passo (T no tempo t0), que ja e conhecida:

          T(t0+dt) = T(t0) + (dt/C) * [Rn(T0) - G(T0) - H(T0) - LE(T0)]

      E um calculo direto (nao precisa resolver nada), mas so e
      numericamente ESTAVEL se dt for pequeno o suficiente frente a
      capacidade termica C e as sensibilidades dos fluxos a T. Como
      C e pequeno (poucos mm de "pele" dossel+solo) e os fluxos de
      calor sensivel/latente sao MUITO sensiveis a T quando ra e
      pequeno, o passo de tempo tipico de 1 hora usado neste modelo
      EXTRAPOLA em muito o limite de estabilidade do metodo explicito
      (ver funcao passo_maximo_estavel() e o script
      comparar_explicito_implicito.py). Isso reproduz exatamente o
      problema que motiva o uso do metodo implicito nos slides.

  (B) METODO IMPLICITO (Euler regressivo / "backward Euler",
      linearizado por serie de Taylor de 1a ordem - IGUAL AO SLIDE)
      -> funcao passo_implicito()
      Os fluxos do lado direito sao avaliados com a temperatura DO
      FIM do passo (T no tempo t0+dt), que ainda e desconhecida. Para
      nao ter que resolver uma equacao nao linear a cada passo,
      expande-se cada fluxo em serie de Taylor de 1a ordem em torno de
      T0:

          f(T0+DT) ~= f(T0) + (df/dT)|_T0 * DT ,  DT = T(t0+dt)-T(t0)

      Substituindo na EDO discretizada (C*DT/dt = Rn-G-H-LE, tudo
      avaliado em T0+DT) e isolando DT, obtem-se uma equacao LINEAR
      em DT (uma unica incognita aqui, pois o modelo usa uma unica
      temperatura de pele T; nos slides do SSiB completo, com Tc e Tg
      separados, o mesmo procedimento gera um sistema 2x2). Esse
      metodo e INCONDICIONALMENTE ESTAVEL (para esta equacao linear),
      permitindo usar passos de tempo de 1 hora sem problemas.

As variaveis auxiliares mais lentas (Td = solo profundo, M = agua
interceptada, theta = umidade do solo) sao atualizadas de forma
EXPLICITA em ambos os casos (sub-passo comum, funcao
atualizar_reservatorios()), pois evoluem em escalas de tempo muito
mais lentas que T e a simplificacao nao compromete a estabilidade.
=====================================================================
"""

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTES FISICAS
# ---------------------------------------------------------------------------
SIGMA  = 5.67e-8      # cte. de Stefan-Boltzmann [W/m2/K4]
CPAIR  = 1004.0       # calor especifico do ar [J/kg/K]
RHOAIR = 1.20         # densidade do ar [kg/m3]
HLAT   = 2.50e6       # calor latente de vaporizacao [J/kg]
PSUR   = 1000.0e2     # pressao de superficie [Pa]
KVK    = 0.41         # constante de von Karman (k)
TF     = 273.15
GAMA   = CPAIR * PSUR / (0.622 * HLAT)   # constante psicrometrica [Pa/K] (gamma = cp*P/(0.622 Le))

# limites de seguranca numerica p/ o metodo explicito (que pode divergir)
T_MIN_SEGURANCA = 150.0    # K
T_MAX_SEGURANCA = 400.0    # K


def esat(T):
    """Pressao de vapor de saturacao es(T) [Pa] - formula tipo Tetens (T em K)."""
    Tc = T - TF
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def desat_dT(T):
    """Derivada des/dT [Pa/K] - usada na linearizacao implicita (Taylor)."""
    Tc = T - TF
    es = esat(T)
    return es * 17.67 * 243.5 / (Tc + 243.5) ** 2


# ---------------------------------------------------------------------------
# PARAMETROS DO MODELO
# ---------------------------------------------------------------------------
class Parametros:
    def __init__(self):
        # --- vegetacao / dossel ---
        self.LAI   = 3.0        # L = indice de area foliar [m2/m2]
        self.V     = 0.85       # fracao vegetada da superficie
        self.hc    = 15.0       # altura do dossel [m] (floresta tropical)
        # NOTA IMPORTANTE: o slide "Fluxo de momentum com vegetacao" traz,
        # para floresta tropical, z0~0.9*hc e d~0.1*hc. Esses valores,
        # aplicados literalmente na formula ra=[ln((z-d)/z0)/(k u)]^2,
        # tornam (z-d)/z0 muito proximo de 1 (ln->0) e ra praticamente nulo,
        # o que e fisicamente inconsistente (ar "deslizaria" livremente sobre
        # o dossel) e causa explosao numerica no esquema implicito. A
        # literatura padrao para florestas altas (Arya; Garratt; Monteith &
        # Unsworth) usa a convencao OPOSTA: d/hc ~ 0.6-0.8 (grande fracao da
        # altura, pois o nivel de referencia do escoamento e deslocado para
        # perto do topo do dossel) e z0/hc ~ 0.05-0.10 (pequena fracao,
        # rugosidade aerodinamica). Adotamos aqui essa convencao padrao -
        # recomenda-se checar com o professor se os valores do slide
        # (0,9 e 0,1) nao estao trocados entre si.
        self.z0    = 0.10 * self.hc  # comprimento de rugosidade
        self.d     = 0.70 * self.hc  # deslocamento do plano zero
        self.zr    = self.hc + 10.0  # altura de referencia das observacoes [m]
        self.ra_min = 5.0            # piso fisico de seguranca para ra [s/m]

        # --- albedo e emissividade (slide: floresta tropical) ---
        self.alb_c = 0.18       # alpha_c: albedo da fracao vegetada
        self.alb_s = 0.15       # alpha_s: albedo do solo exposto
        self.eps_c = 0.98       # eps_c: emissividade da vegetacao
        self.eps_s = 0.94       # eps_s: emissividade do solo exposto

        # --- interceptacao (M, S=capacidade maxima de agua na folha) ---
        self.S_por_LAI = 0.2    # S proporcional a L  ->  S = S_por_LAI * LAI [mm]

        # --- resistencias ---
        self.rs_min = 100.0     # resistencia estomatica minima [s/m] (estomato aberto)
        self.rs_max = 5000.0    # resistencia estomatica maxima [s/m] (estomato fechado)
        self.RG_ref = 100.0     # radiacao de referencia p/ resposta estomatica a luz [W/m2]
        self.rsoil_min = 150.0  # resistencia minima de evaporacao do solo [s/m]

        # --- capacidade termica efetiva do sistema (slide: C~1e3 a 1e4 J/K/m2) ---
        self.C  = 5.0e3         # capacidade termica da "pele" (dossel+solo) [J/m2/K]
        self.Cd = 2.0e6         # capacidade termica do solo profundo [J/m2/K]
        self.G_lambda = 6.0     # condutividade termica efetiva do solo [W/m/K]
        self.G_depth  = 0.10    # profundidade efetiva entre T (pele) e Td [m]

        # --- reservatorio de agua no solo (balanco hidrico) ---
        self.D          = 500.0   # D: profundidade da camada de solo [mm]
        self.theta_sat  = 0.42    # umidade de saturacao (porosidade)
        self.theta_wilt = 0.10    # ponto de murcha (umidade minima disponivel)
        self.Ru_max     = 4.0 / 86400.0   # drenagem profunda maxima (solo saturado) [mm/s] (~4 mm/dia)


# ---------------------------------------------------------------------------
# FORCANTES SINTETICAS - CICLO DIURNO DE 72 HORAS (3 DIAS)
# ---------------------------------------------------------------------------
def gerar_forcante_sintetica(nhoras=72, dt=3600.0, chuva_dia=2, seed=42):
    """
    Series sinteticas horarias representando um ciclo diurno idealizado,
    repetido por 3 dias: SWd (radiacao de onda curta incidente), LWd (onda
    longa incidente), Tr (temperatura do ar no nivel de referencia), er
    (pressao de vapor do ar), Ur (vento no nivel de referencia) e P0
    (precipitacao no topo do dossel). Um evento de chuva sintetico e
    inserido no `chuva_dia` para ilustrar a dinamica de interceptacao e
    do reservatorio de agua no solo.
    """
    rng = np.random.default_rng(seed)
    n = int(nhoras * 3600 / dt) + 1
    t = np.arange(n) * dt
    horas = t / 3600.0
    hora_do_dia = horas % 24.0

    # --- SWd: radiacao de onda curta incidente ---
    SWd_max = 850.0
    fase_solar = np.clip(np.sin(np.pi * (hora_do_dia - 6.0) / 12.0), 0, None)
    nublado = np.where((horas >= 24) & (horas < 48), 0.55, 1.0)   # dia 2 nublado
    SWd = SWd_max * fase_solar * nublado
    SWd += rng.normal(0, 5, n) * (SWd > 0)
    SWd = np.clip(SWd, 0, None)

    # --- Tr: temperatura do ar no nivel de referencia ---
    Tr_media, Tr_amp = 296.0, 6.0
    Tr = Tr_media + Tr_amp * np.sin(2 * np.pi * (hora_do_dia - 9.0) / 24.0)
    Tr -= 1.5 * (1 - nublado)
    Tr += rng.normal(0, 0.2, n)

    # --- er: pressao de vapor do ar (via UR sintetica) ---
    RH = np.clip(0.55 + 0.30 * np.sin(2 * np.pi * (hora_do_dia - 21.0) / 24.0), 0.25, 0.98)
    er = RH * esat(Tr)

    # --- Ur: vento no nivel de referencia ---
    Ur = 2.5 + 1.8 * np.clip(np.sin(np.pi * (hora_do_dia - 6.0) / 12.0), 0, None)
    Ur += rng.normal(0, 0.15, n)
    Ur = np.clip(Ur, 0.3, None)

    # --- LWd: radiacao de onda longa atmosferica (Brutsaert simplificado) ---
    eps_atm = np.clip(0.70 + 0.20 * (er / 1000.0) ** 0.25, 0.6, 0.98)
    eps_atm = np.where((horas >= 24) & (horas < 48), np.clip(eps_atm + 0.15, 0, 1), eps_atm)
    LWd = eps_atm * SIGMA * Tr ** 4

    # --- P0: precipitacao sintetica no topo do dossel ---
    P0 = np.zeros(n)   # mm/s
    ini = int((24 * (chuva_dia - 1) + 14) * 3600 / dt)
    fim = int((24 * (chuva_dia - 1) + 16) * 3600 / dt)
    if 0 <= ini < n:
        P0[ini:min(fim, n)] = 6.0 / 3600.0    # ~6 mm/h por 2h -> total ~12 mm

    return dict(t=t, horas=horas, dt=dt, SWd=SWd, LWd=LWd, Tr=Tr, er=er,
                Ur=Ur, P0=P0, RH=RH)


# ---------------------------------------------------------------------------
# CALCULO DOS FLUXOS (comum aos dois metodos) + DERIVADAS (so p/ o implicito)
# ---------------------------------------------------------------------------
def calcular_fluxos(T, Td, M, theta, forc_i, p):
    """
    Calcula todas as resistencias e fluxos de energia do sistema
    solo-planta-atmosfera para uma dada temperatura T (podendo ser a
    temperatura do INICIO do passo, no metodo explicito, ou uma
    temperatura de referencia T0 para a linearizacao, no metodo
    implicito). Retorna tambem as derivadas parciais df/dT de cada
    fluxo, usadas apenas pelo metodo implicito.
    """
    SWd, LWd, Tr, er, Ur, P0 = forc_i
    Ur = max(Ur, 0.3)

    # ---------------- 1) FLUXO DE MOMENTUM / RESISTENCIA AERODINAMICA ------
    # ATUALIZACAO (slide "Estrutura do Modelo", pagina nova - modelo de
    # Verma-Rosenberg): a forma padrao usa o coeficiente de arrasto (CD)
    # explicitamente, com ra = 1/(CD*U) - ou seja, um UNICO fator de U no
    # denominador (nao U ao quadrado). A formula anteriormente usada aqui
    # ("1/ra = [k*u/ln(...)]^2", copiada literalmente do vegetacao.pdf)
    # colocava U DENTRO do colchete elevado ao quadrado, o que produz ra
    # decrescendo com 1/U^2 (fisicamente incorreto e numericamente instavel
    # para U mais alto). A forma correta e padrao de livro-texto e:
    #     CD_neutro = [k / ln((zr-d)/z0)]^2      (adimensional)
    #     ra = 1 / (CD_neutro * U)                (s/m)
    ln_term = np.log((p.zr - p.d) / p.z0)
    CD_neutro = (KVK / ln_term) ** 2
    ra = 1.0 / (CD_neutro * Ur)
    ra = max(ra, p.ra_min)                     # piso fisico de seguranca
    tau = RHOAIR * CD_neutro * Ur ** 2          # fluxo de momentum (diagnostico)

    # ---------------- 2) INTERCEPTACAO: W = M/S, S proporcional a L --------
    S = p.S_por_LAI * p.LAI
    W = np.clip(M / S, 0.0, 1.0) if S > 0 else 0.0

    # ---------------- 3) RESISTENCIA ESTOMATICA E DE COPA -------------------
    # rs (Jarvis simplificado: luz + umidade do solo) ; rc = rs / LAI
    theta_disp = np.clip((theta - p.theta_wilt) / (p.theta_sat - p.theta_wilt), 0.02, 1.0)
    f_luz  = SWd / (SWd + p.RG_ref) if SWd > 0 else 0.0
    f_solo = theta_disp
    rs = p.rs_min / max(f_luz * f_solo, 1e-3)
    rs = min(rs, p.rs_max) if SWd > 0 else p.rs_max
    rc = rs / p.LAI

    # resistencia do solo nu (aumenta muito quando o solo seca)
    rsoil = p.rsoil_min / theta_disp ** 0.7

    esT = esat(T)
    desT = desat_dT(T)

    # ---------------- 4) SALDO DE RADIACAO (slide "saldo de radiacao") ------
    alb_eff = p.V * p.alb_c + (1.0 - p.V) * p.alb_s
    eps_eff = p.V * p.eps_c + (1.0 - p.V) * p.eps_s
    Rn0 = (1.0 - alb_eff) * SWd + LWd - eps_eff * SIGMA * T ** 4
    dRn_dT = -4.0 * eps_eff * SIGMA * T ** 3

    # ---------------- 5) CALOR SENSIVEL (H = rho*cp*(T-Tr)/ra) -------------
    H0 = RHOAIR * CPAIR * (T - Tr) / ra
    dH_dT = RHOAIR * CPAIR / ra

    # ---------------- 6) CALOR LATENTE: LEi, LEc, LEs -----------------------
    # LEi = W * (rho cp/gamma) * (es(T)-er)/ra           [agua interceptada]
    # LEc = (1-W) * (rho cp/gamma) * (es(T)-er)/(ra+rc)   [transpiracao]
    # LEs = beta(theta) * (rho cp/gamma) * (es(T)-er)/(ra+rsoil) [solo nu]
    Ki = RHOAIR * CPAIR / GAMA * W / ra
    Kc = RHOAIR * CPAIR / GAMA * (1.0 - W) / (ra + rc)
    beta_solo = theta_disp   # fator de limitacao da evaporacao do solo
    Ks = RHOAIR * CPAIR / GAMA * beta_solo * (1.0 - p.V) / (ra + rsoil)

    LEi0 = Ki * (esT - er)
    LEc0 = Kc * (esT - er)
    LEs0 = Ks * (esT - er)
    dLEi_dT = Ki * desT
    dLEc_dT = Kc * desT
    dLEs_dT = Ks * desT

    # ---------------- 7) FLUXO DE CALOR NO SOLO (force-restore simples) ----
    Gflux0 = p.G_lambda / p.G_depth * (T - Td)
    dG_dT = p.G_lambda / p.G_depth

    return dict(
        ra=ra, tau=tau, W=W, S=S, rs=rs, rc=rc, rsoil=rsoil, theta_disp=theta_disp,
        Rn0=Rn0, dRn_dT=dRn_dT, H0=H0, dH_dT=dH_dT,
        LEi0=LEi0, dLEi_dT=dLEi_dT, LEc0=LEc0, dLEc_dT=dLEc_dT,
        LEs0=LEs0, dLEs_dT=dLEs_dT, Gflux0=Gflux0, dG_dT=dG_dT,
    )


def atualizar_reservatorios(Td, M, theta, dt, p, Gflux, LEi, LEc, LEs, H, S,
                             P0):
    """
    Atualizacao EXPLICITA (Euler progressivo) dos reservatorios "lentos":
    Td (solo profundo, force-restore), M (agua interceptada) e theta
    (umidade do solo). Usada igualmente pelos metodos explicito e
    implicito, pois estas variaveis evoluem devagar e a simplificacao
    explicita nao compromete a estabilidade do esquema.
    Retorna: Td_novo, M_novo, theta_novo, Ei, Ec, Es, P1, Rs, Ru, H_corrigido,
             LEi_corrigido
    """
    # solo profundo: force-restore simples, EXPLICITO
    Td_novo = Td + dt * Gflux / p.Cd

    # ---------------- BALANCO HIDRICO (EXPLICITO) ---------------------------
    # dM/dt = (P0 - P1) - Ei   (Ei em mm/s = LEi/L)
    Ei = LEi / HLAT
    Ec = LEc / HLAT
    Es = LEs / HLAT

    # capacidade de armazenamento restante na copa determina o throughfall
    cap_livre = max(S - M, 0.0) / dt
    P1 = np.maximum(0.0, P0 - cap_livre)          # agua que atravessa o dossel

    # --- correcao de conservacao de massa: a evaporacao da agua interceptada
    # nao pode exceder a agua efetivamente disponivel na copa (M + chuva
    # recebida no passo). O excesso de energia que nao pode ser usado para
    # evaporar e redirecionado para o calor sensivel (igual a correcao
    # ECIDIF da subrotina TEMRS1/UPDAT1 do SSiB). Isso evita picos irreais
    # de LEi logo apos a saturacao do dossel pela chuva. IMPORTANTE: LEi
    # tambem precisa ser reduzido (nao apenas H aumentado), senao o balanco
    # de energia deixa de fechar.
    M_disponivel = M + (P0 - P1) * dt
    Ei_max = M_disponivel / dt
    if Ei > Ei_max:
        excesso_LE = (Ei - Ei_max) * HLAT
        Ei = Ei_max
        LEi = LEi - excesso_LE
        H = H + excesso_LE

    dM = (P0 - P1 - Ei) * dt
    M_novo = np.clip(M + dM, 0.0, S)

    # D dtheta/dt = P1 - (Ec+Es) - (Rs+Ru)
    theta_disp_local = np.clip((theta - p.theta_wilt) / (p.theta_sat - p.theta_wilt), 0.02, 1.0)
    Ru = p.Ru_max * theta_disp_local ** 4
    infiltra_pot = P1                              # tenta infiltrar todo o throughfall
    theta_tent = theta + (infiltra_pot - Ec - Es - Ru) * dt / p.D
    Rs = np.maximum(0.0, (theta_tent - p.theta_sat)) * p.D / dt   # escoamento superficial
    theta_novo = np.clip(theta_tent - Rs * dt / p.D, p.theta_wilt * 0.5, p.theta_sat)

    return Td_novo, M_novo, theta_novo, Ei, Ec, Es, P1, Rs, Ru, H, LEi


# ---------------------------------------------------------------------------
# (A) METODO EXPLICITO (Euler progressivo) para a temperatura T
# ---------------------------------------------------------------------------
def passo_explicito(state, forc_i, p, dt):
    """
    Avanca T de forma EXPLICITA:
        T(t0+dt) = T(t0) + (dt/C) * [Rn(T0) - G(T0) - H(T0) - LE(T0)]
    isto e, todos os fluxos do lado direito usam a temperatura T0 JA
    CONHECIDA no inicio do passo - nao ha necessidade de resolver
    nenhuma equacao, mas o esquema so e estavel para dt pequeno.
    """
    T, Td, M, theta = state
    SWd, LWd, Tr, er, Ur, P0 = forc_i

    fx = calcular_fluxos(T, Td, M, theta, forc_i, p)

    LE0_total = fx['LEi0'] + fx['LEc0'] + fx['LEs0']
    # ---- Euler progressivo (EXPLICITO): usa somente valores em T0 --------
    T_novo = T + (dt / p.C) * (fx['Rn0'] - fx['Gflux0'] - fx['H0'] - LE0_total)

    # protecao numerica: o metodo explicito pode divergir (esse e o ponto!)
    # sem este limite, T^4 na formula de Rn geraria overflow em poucos passos
    T_novo = np.clip(T_novo, T_MIN_SEGURANCA, T_MAX_SEGURANCA)

    Td_novo, M_novo, theta_novo, Ei, Ec, Es, P1, Rs, Ru, H_corr, LEi_corr = \
        atualizar_reservatorios(Td, M, theta, dt, p, fx['Gflux0'],
                                 fx['LEi0'], fx['LEc0'], fx['LEs0'],
                                 fx['H0'], fx['S'], P0)

    novo_estado = np.array([T_novo, Td_novo, M_novo, theta_novo])
    diag = dict(metodo='explicito', ra=fx['ra'], tau=fx['tau'], W=fx['W'],
                rs=fx['rs'], rc=fx['rc'], rsoil=fx['rsoil'],
                Rn=fx['Rn0'], H=H_corr, LEi=LEi_corr, LEc=fx['LEc0'],
                LEs=fx['LEs0'], G=fx['Gflux0'], Ei=Ei, Ec=Ec, Es=Es,
                P1=P1, Rs=Rs, Ru=Ru, theta_disp=fx['theta_disp'])
    return novo_estado, diag


def passo_maximo_estavel(state, forc_i, p):
    """
    Estima o passo de tempo maximo dt_max para o qual o metodo EXPLICITO
    permanece estavel, a partir do criterio classico de Euler progressivo
    para uma EDO linear dT/dt = -(A/C)*T + ... :  dt_max = 2*C/A , onde
    A = soma das sensibilidades (dH/dT + dLE/dT + dG/dT - dRn/dT).
    Uteis para comparar com o dt=3600 s usado na simulacao.
    """
    T, Td, M, theta = state
    fx = calcular_fluxos(T, Td, M, theta, forc_i, p)
    A = (-fx['dRn_dT'] + fx['dG_dT'] + fx['dH_dT']
         + fx['dLEi_dT'] + fx['dLEc_dT'] + fx['dLEs_dT'])
    return 2.0 * p.C / A if A > 0 else np.inf


# ---------------------------------------------------------------------------
# (B) METODO IMPLICITO (Euler regressivo, linearizado por Taylor) - SLIDE
# ---------------------------------------------------------------------------
def passo_implicito(state, forc_i, p, dt):
    """
    Avanca T de forma IMPLICITA (backward Euler), linearizando cada fluxo
    em serie de Taylor de 1a ordem em torno de T0 (EXATAMENTE o metodo
    descrito no slide "Solucao Numerica das Equacoes Prognosticas"):

        f(T0+DT) ~= f(T0) + (df/dT)|_T0 * DT

    Substituindo em  C*DT/dt = Rn(T0+DT) - G(T0+DT) - H(T0+DT) - LE(T0+DT)
    e isolando DT, obtem-se uma equacao linear (aqui escalar, pois o
    modelo tem uma unica temperatura de pele T):

        [C/dt + (dH/dT+dLE/dT+dG/dT-dRn/dT)] * DT = Rn0-G0-H0-LE0

    Este esquema e incondicionalmente estavel para a parte linearizada,
    permitindo dt grandes (ex.: 1 hora) sem oscilacao numerica.
    """
    T, Td, M, theta = state
    SWd, LWd, Tr, er, Ur, P0 = forc_i

    fx = calcular_fluxos(T, Td, M, theta, forc_i, p)

    # ---- montagem do sistema LINEAR (escalar) para DT ---------------------
    A = (p.C / dt
         + (-fx['dRn_dT'] + fx['dG_dT'] + fx['dH_dT']
            + fx['dLEi_dT'] + fx['dLEc_dT'] + fx['dLEs_dT']))
    RHS = fx['Rn0'] - fx['Gflux0'] - fx['H0'] - (fx['LEi0'] + fx['LEc0'] + fx['LEs0'])
    DT = RHS / A
    T_novo = T + DT

    # fluxos finais (1a ordem em torno de T0, avaliados em T0+DT)
    Rn    = fx['Rn0']   + fx['dRn_dT']   * DT
    H     = fx['H0']    + fx['dH_dT']    * DT
    LEi   = fx['LEi0']  + fx['dLEi_dT']  * DT
    LEc   = fx['LEc0']  + fx['dLEc_dT']  * DT
    LEs   = fx['LEs0']  + fx['dLEs_dT']  * DT
    Gflux = fx['Gflux0']+ fx['dG_dT']    * DT

    Td_novo, M_novo, theta_novo, Ei, Ec, Es, P1, Rs, Ru, H_corr, LEi_corr = \
        atualizar_reservatorios(Td, M, theta, dt, p, Gflux, LEi, LEc, LEs,
                                 H, fx['S'], P0)

    novo_estado = np.array([T_novo, Td_novo, M_novo, theta_novo])
    diag = dict(metodo='implicito', ra=fx['ra'], tau=fx['tau'], W=fx['W'],
                rs=fx['rs'], rc=fx['rc'], rsoil=fx['rsoil'],
                Rn=Rn, H=H_corr, LEi=LEi_corr, LEc=LEc, LEs=LEs, G=Gflux,
                Ei=Ei, Ec=Ec, Es=Es, P1=P1, Rs=Rs, Ru=Ru,
                theta_disp=fx['theta_disp'])
    return novo_estado, diag


# alias mantido por compatibilidade com versoes anteriores do script
# (equivalente a passo_implicito, que e o metodo usado por padrao)
def passo_modelo(state, forc_i, p, dt):
    return passo_implicito(state, forc_i, p, dt)


# ---------------------------------------------------------------------------
# LOOP PRINCIPAL DE SIMULACAO
# ---------------------------------------------------------------------------
def rodar_simulacao(p=None, nhoras=72, dt=3600.0, chuva_dia=2,
                     T0=295.0, Td0=294.0, M0=0.0, theta0=0.28,
                     metodo='implicito'):
    """
    Executa a integracao completa do ciclo diurno de `nhoras` horas.
    metodo: 'implicito' (padrao, igual ao slide, estavel) ou
            'explicito' (Euler progressivo, para fins de comparacao -
            pode divergir/oscilar com dt=3600 s).
    """
    if p is None:
        p = Parametros()
    forc = gerar_forcante_sintetica(nhoras=nhoras, dt=dt, chuva_dia=chuva_dia)
    n = len(forc['t'])

    passo_fn = passo_implicito if metodo == 'implicito' else passo_explicito

    state = np.array([T0, Td0, M0, theta0])
    campos = ['T', 'Td', 'M', 'theta', 'ra', 'tau', 'W', 'rs', 'rc', 'rsoil',
              'Rn', 'H', 'LEi', 'LEc', 'LEs', 'G', 'Ei', 'Ec', 'Es', 'P1',
              'Rs', 'Ru', 'theta_disp']
    hist = {k: np.zeros(n) for k in campos}

    for i in range(n):
        forc_i = (forc['SWd'][i], forc['LWd'][i], forc['Tr'][i], forc['er'][i],
                  forc['Ur'][i], forc['P0'][i])
        state, diag = passo_fn(state, forc_i, p, dt)
        hist['T'][i], hist['Td'][i], hist['M'][i], hist['theta'][i] = state
        for k in campos[4:]:
            hist[k][i] = diag[k]

    return forc, hist, p
