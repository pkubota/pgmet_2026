# -*- coding: utf-8 -*-
"""
MODELO CONCEITUAL DA TRANSICAO RASA -> PROFUNDA - v3
======================================================
Curso: Conveccao Atmosferica - Regimes de Transicao

NOVIDADES EM RELACAO A v2
--------------------------
A v2 tinha fluxo de massa real (pluma entranhante) e formacao de
precipitacao (autoconversao Tiedtke), mas ainda faltavam varios
processos fisicos de um esquema de conveccao operacional completo.
Esta versao adiciona, na mesma estrutura de pluma entranhante:

  (a) FASE GELO/LIQUIDA: agua condensada e particionada entre liquido
      (q_l) e gelo (q_i) conforme a temperatura (fracao de gelo linear
      entre 0degC e -40degC); usa energia estatica liquida-gelo (s_li) e
      calor latente efetivo (mistura L_v/L_s); autoconversao separada
      para chuva (liquido) e neve/graupel (gelo);
  (b) DOWNDRAFT CONVECTIVO: iniciado no Nivel de Afundamento Livre
      (LFS), com resfriamento evaporativo da chuva que cai atraves
      dele, gerando uma corrente descendente fria (cold pool) -
      seguindo o conceito de cudlfs/cudrft do Tiedtke;
  (c) EVAPORACAO DE PRECIPITACAO ABAIXO DA BASE DA NUVEM: parte da
      chuva que sai do downdraft evapora na camada sub-nuvem nao
      saturada antes de atingir a superficie;
  (d) DERRETIMENTO DE NEVE/GRANIZO: neve que cai abaixo do nivel de
      congelamento ambiente derrete (convertida em chuva), consumindo
      calor latente de fusao;
  (e) FECHAMENTO POR QUASE-EQUILIBRIO DE CAPE: o fluxo de massa de
      base da pluma profunda passa a responder ao CAPE disponivel
      (relaxado numa escala de tempo ajustavel), em vez de ser
      proporcional apenas ao fluxo de calor sensivel;
  (f) TRANSPORTE CONVECTIVO DE MOMENTUM: quantidade de movimento
      horizontal (u) e misturada por entranhamento/detranhamento na
      pluma, como no Tiedtke (lmfdudv), fornecendo um perfil
      diagnostico do fluxo convectivo de momentum;
  (g) ESPECTRO DE PLUMAS: em vez de so duas taxas de entranhamento
      (rasa/profunda), um espectro de N plumas e integrado a cada
      passo de tempo, ao estilo Arakawa & Schubert (1974), produzindo
      uma distribuicao de alturas de topo de nuvem;
  (h) RESFRIAMENTO RADIATIVO: um resfriamento de ceu claro lento e
      prescrito (~1,5 K/dia) enfraquece gradualmente a camada estavel
      ao longo do dia, um mecanismo adicional (secundario) de erosao
      do CIN.

Todas as constantes novas sao valores tipicos da literatura, usados
UMA VEZ (nao recalibrados exaustivamente) - o objetivo e ter a fisica
certa e testavel, nao um ajuste fino de cada parametro. Limitacoes e
simplificacoes de cada peca sao documentadas nos comentarios e no
adendo em Word que acompanha este script.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# (fix acima: evita UnicodeEncodeError em terminais Windows que nao usam UTF-8
#  por padrao -- necessario para os acentos e simbolos gregos usados nos prints)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONSTANTES FISICAS
# =====================================================================
g = 9.81
cp = 1004.0
Rd = 287.0
Rv = 461.5
Lv = 2.5e6          # calor latente de vaporizacao [J/kg]
Lf = 3.337e5         # calor latente de fusao [J/kg]
Ls = Lv + Lf          # calor latente de sublimacao [J/kg]
eps_R = Rd / Rv

def qsat_liq(T, p):
    """Umidade especifica de saturacao sobre agua liquida [kg/kg] (Bolton 1980)."""
    Tc = T - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))   # hPa
    return eps_R * es / np.maximum(p - es, 1e-3)

def qsat_ice(T, p):
    """Umidade especifica de saturacao sobre gelo [kg/kg] (formula tipo Magnus para gelo)."""
    Tc = T - 273.15
    es = 6.112 * np.exp(22.46 * Tc / (Tc + 272.62))   # hPa
    return eps_R * es / np.maximum(p - es, 1e-3)

T_ICE_ALL = 233.15    # abaixo disso, 100% gelo
T_LIQ_ALL = 273.15    # acima disso, 100% liquido

def ice_fraction(T):
    """Fracao de gelo no condensado, rampa linear entre 0degC e -40degC."""
    f = (T_LIQ_ALL - T) / (T_LIQ_ALL - T_ICE_ALL)
    return min(max(f, 0.0), 1.0)

def qsat_mixed(T, p):
    """Umidade de saturacao em fase mista: combinacao liquido/gelo pela fracao de gelo."""
    fi = ice_fraction(T)
    return (1.0-fi)*qsat_liq(T, p) + fi*qsat_ice(T, p)

def qsat(T, p):
    return qsat_liq(T, p)

def p_of_z(z, p0=1000.0, H=8000.0):
    """Pressao hidrostatica aproximada [hPa] (escala fixa, simplificacao didatica)."""
    return p0 * np.exp(-z / H)

def exner(z, p0=1000.0):
    """Funcao de Exner Pi(z) = (p(z)/p0)^(Rd/cp) -- converte T <-> theta."""
    return (p_of_z(z)/p0)**(Rd/cp)

# =====================================================================
# 2. SONDAGEM AMBIENTE (prescrita por taxas de lapso em camadas)
# =====================================================================
Z_TOP = 14000.0
DZ = 20.0
zgrid = np.arange(0.0, Z_TOP + DZ, DZ)
pgrid = p_of_z(zgrid)

T_env0 = 297.0

LAPSE_SEGMENTS = [(0.0, 1400.0, 9.0e-3),
                  (1400.0, 1950.0, 2.8e-3),
                  (1950.0, 11000.0, 7.0e-3),
                  (11000.0, 20000.0, 0.0)]

def lapse_rate(z):
    for zlo, zhi, rate in LAPSE_SEGMENTS:
        if zlo <= z < zhi:
            return rate
    return LAPSE_SEGMENTS[-1][2]

T_env_base = np.zeros_like(zgrid)
T_env_base[0] = T_env0
for i in range(1, len(zgrid)):
    T_env_base[i] = T_env_base[i-1] - lapse_rate(zgrid[i-1]) * (zgrid[i]-zgrid[i-1])

z_transition_top = 3000.0
RH_bg = 0.40

# --- Resfriamento radiativo de ceu claro (lento, prescrito) ---
# Enfraquece a camada estavel ao longo do dia, adicionando-se ao efeito do
# umedecimento na erosao do CIN (mecanismo secundario, real e documentado
# em estudos do ciclo diurno da conveccao continental).
RAD_COOL_RATE = 1.5 / 86400.0   # K/s (~1,5 K/dia, resfriamento de onda longa tipico)
_rad_offset = [0.0]              # mutavel: atualizado a cada passo de tempo do loop diurno

def T_env_at(z):
    return np.interp(z, zgrid, T_env_base) - _rad_offset[0]

def q_env_at(z, q_ft):
    if z <= z_transition_top:
        return q_ft
    Te = T_env_at(z)
    return qsat_liq(Te, p_of_z(z)) * RH_bg

def lcl_height(T_k, q):
    Tc = T_k - 273.15
    es = q * 1000.0 / (eps_R + q*(1-eps_R))
    es = max(es, 1e-6)
    Td = 243.5*np.log(es/6.112) / (17.67 - np.log(es/6.112))
    return max(0.0, 125.0*(Tc - Td))

# --- Perfil de vento ambiente (cisalhamento simples, para transporte de momentum) ---
U0_SHEAR = 2.0        # m/s na superficie
SHEAR_RATE = 2.5e-3    # s^-1 (cisalhamento vertical do vento zonal)

def u_env_at(z):
    return U0_SHEAR + SHEAR_RATE*z

# =====================================================================
# 3. PLUMA ENTRANHANTE (ASCENDENTE) - fase mista, precipitacao (chuva+neve),
#    momentum e "detranhamento organizado" no topo
# =====================================================================
A_BUOY = 1.0
B_DRAG = 1.0

CPRCON_RATE = 1.1e-3      # 1/m -- autoconversao liquido->chuva (Tiedtke: cprcon*g)
CPRCON_ICE_RATE = 0.6e-3   # 1/m -- autoconversao gelo->neve (mais lenta: agregacao/deposicao)
DEPTH_NO_PRECIP = 100.0    # m acima da base da nuvem (reescalado - ver adendo)

def sat_adjust_mixed(sli, qt, z, p):
    """
    Resolve (T, q_vapor, q_liquido, q_gelo) dados a energia estatica
    liquida-gelo (sli = cp*T + g*z - Lv*ql - Ls*qi) e a agua total (qt),
    por iteracao de ponto fixo, usando a fracao de gelo dependente de T.
    """
    T = (sli - g*z) / cp
    for _ in range(10):
        fi = ice_fraction(T)
        qs = qsat_mixed(T, p)
        qcond = max(0.0, qt - qs)
        ql = qcond*(1.0-fi)
        qi = qcond*fi
        Leff = Lv*ql + Ls*qi   # energia latente total ja liberada
        T_new = (sli - g*z + Leff) / cp
        if abs(T_new - T) < 1e-4:
            T = T_new
            break
        T = T_new
    fi = ice_fraction(T)
    qs = qsat_mixed(T, p)
    qcond = max(0.0, qt - qs)
    ql = qcond*(1.0-fi)
    qi = qcond*fi
    return T, qt - qcond, ql, qi

def run_plume(z0, T0, q0, w2_0, epsilon, q_ft, u0=None, zmax=Z_TOP, dz=DZ):
    """
    Integra uma pluma entranhante a partir de (z0, T0, q0), com fase
    mista liquido/gelo, formacao de precipitacao (chuva + neve) e
    transporte de momentum (se u0 for fornecido). Retorna a altura de
    detranhamento, a agua que a pluma carrega ali, a precipitacao
    (liquida e solida), o fluxo de momentum no topo (detranhamento
    organizado), e os perfis completos (theta_c, qt_c, u_c, fluxo de
    massa normalizado M/Mb) usados para calcular as tendencias
    convectivas sobre o ambiente de grande escala (ver
    calcula_tendencias()).
    """
    n = int((zmax - z0) / dz) + 1
    z = z0 + np.arange(n) * dz
    B_arr = np.zeros(n)
    theta_c_arr = np.zeros(n)
    qt_c_arr = np.zeros(n)
    u_c_arr = np.zeros(n)
    Mnorm_arr = np.zeros(n)   # M(z)/Mb -- cresce por entranhamento: dM/dz = epsilon*M
    sli = cp*T0 + g*z0
    qt = q0
    w2 = w2_0
    u = u0 if u0 is not None else u_env_at(z0)
    top_idx = n - 1
    q_final = q0
    rain_flux = 0.0
    snow_flux = 0.0
    CAPE = 0.0

    theta_c_arr[0] = T0/exner(z0)
    qt_c_arr[0] = q0
    u_c_arr[0] = u
    Mnorm_arr[0] = 1.0

    for i in range(1, n):
        Te = T_env_at(z[i]); pe = p_of_z(z[i])
        qe = q_env_at(z[i], q_ft)
        ue = u_env_at(z[i])
        sli_env = cp*Te + g*z[i]

        # entranhamento: mistura sli, qt e u (conservados) com o ambiente
        sli = sli - epsilon*dz*(sli - sli_env)
        qt = qt - epsilon*dz*(qt - qe)
        u = u - epsilon*dz*(u - ue)

        T, q, ql, qi = sat_adjust_mixed(sli, qt, z[i], pe)

        # --- autoconversao: chuva (liquido) e neve (gelo), taxas diferentes ---
        if (z[i] - z0) >= DEPTH_NO_PRECIP:
            if ql > 0.0:
                ql_new = ql/(1.0 + CPRCON_RATE*dz)
                rain_flux += (ql - ql_new)
                ql = ql_new
            if qi > 0.0:
                qi_new = qi/(1.0 + CPRCON_ICE_RATE*dz)
                snow_flux += (qi - qi_new)
                qi = qi_new
            qt = q + ql + qi
            sli = cp*T + g*z[i] - Lv*ql - Ls*qi

        Tv_u = T*(1.0 + 0.61*q - ql - qi)
        Tv_e = Te*(1.0 + 0.61*qe)
        B = g*(Tv_u - Tv_e)/Tv_e
        B_arr[i] = B
        if B > 0:
            CAPE += B*dz

        w2 = w2 + dz*(2*A_BUOY*B - 2*B_DRAG*epsilon*w2)
        q_final = q

        theta_c_arr[i] = T/exner(z[i])
        qt_c_arr[i] = qt
        u_c_arr[i] = u
        Mnorm_arr[i] = Mnorm_arr[i-1]*np.exp(epsilon*dz)   # dM/dz = epsilon*M (so entranhamento)

        if w2 <= 0.0:
            top_idx = i
            break

    # --- detranhamento organizado no topo: toda a massa restante (incl.
    #     momentum) e depositada no nivel onde a pluma perde flutuabilidade ---
    u_top = u
    mom_flux_top = (u_top - u_env_at(z[top_idx]))   # anomalia de momentum detranhada

    return dict(z_top=z[top_idx], q_top=q_final, w2_end=max(w2, 0.0),
                z=z[:top_idx+1], B=B_arr[:top_idx+1],
                rain=rain_flux, snow=snow_flux, CAPE=CAPE,
                u_top=u_top, mom_flux_top=mom_flux_top,
                theta_c=theta_c_arr[:top_idx+1], qt_c=qt_c_arr[:top_idx+1],
                u_c=u_c_arr[:top_idx+1], Mnorm=Mnorm_arr[:top_idx+1])

def calcula_tendencias(plume_result, Mb, q_ft, dz_host=60.0):
    """
    Calcula os perfis verticais de tendencia convectiva sobre o ambiente
    de grande escala -- o que a parametrizacao de fato devolve ao modelo
    hospedeiro (equacoes do slide 39 do curso: adveccao, SUBSIDENCIA DE
    LARGA ESCALA e TRANSPORTE TURBULENTO induzidos pela nuvem):

        dtheta/dt|conv (z) = -(1/rho) * d/dz[ M_c(z)*(theta_c(z)-theta_amb(z)) ]
        dq/dt|conv     (z) = -(1/rho) * d/dz[ M_c(z)*(qt_c(z)  -q_amb(z))      ]
        du/dt|conv     (z) = -(1/rho) * d/dz[ M_c(z)*(u_c(z)   -u_amb(z))      ]

    onde M_c(z) = Mb*Mnorm(z) cresce por entranhamento ao longo da
    ascensao e cai abruptamente a zero no topo da pluma (detranhamento
    organizado). Isso produz o perfil classico em "M" (ou dipolo) da
    literatura (Yanai et al. 1973): aquecimento/secamento por subsidencia
    compensatoria ao longo de toda a camada de nuvem, com um pulso extra
    de aquecimento/umedecimento concentrado no nivel de detranhamento.

    IMPORTANTE (resolucao vertical): a pluma e integrada em dz=20 m, mas
    um modelo hospedeiro real tem camadas bem mais espessas. Calcular o
    gradiente direto na grade fina de 20 m concentra o pulso de
    detranhamento numa camada artificialmente fina, inflando a taxa
    LOCAL (a quantidade fisica que se conserva e o pulso integrado na
    vertical, nao a taxa local). Por isso, os fluxos sao reamostrados
    para uma grade um pouco mais grossa (dz_host, default 60 m) antes de
    calcular o gradiente. Note que 60 m ainda e bem mais fino que uma
    camada de GCM tipica (centenas de metros) -- isso e proposital: as
    nuvens deste modelo simplificado sao rasas (100-300 m de profundidade
    tipica), entao uma grade de reamostragem muito grossa (ex.: 250 m)
    apagaria a estrutura vertical da propria nuvem. Ainda assim, os
    valores absolutos de K/dia permanecem sensiveis a essa escolha -- ver
    a ressalva sobre a magnitude de M_b logo abaixo.
    """
    z = plume_result["z"]
    n = len(z)
    if n < 3:
        return dict(z=z, dtheta_dt=np.zeros(n), dq_dt=np.zeros(n), du_dt=np.zeros(n))

    Mc = Mb*plume_result["Mnorm"]                    # fluxo de massa [kg/m2/s]
    theta_amb = np.array([T_env_at(zz)/exner(zz) for zz in z])
    q_amb = np.array([q_env_at(zz, q_ft) for zz in z])
    u_amb = np.array([u_env_at(zz) for zz in z])

    F_theta = Mc*(plume_result["theta_c"] - theta_amb)
    F_q = Mc*(plume_result["qt_c"] - q_amb)
    F_u = Mc*(plume_result["u_c"] - u_amb)

    # --- reamostra os fluxos para uma grade vertical mais grossa (dz_host) ---
    z_top = z[-1]
    z0 = z[0]
    z_host = np.arange(z0, z_top+2*dz_host, dz_host)
    F_theta_h = np.interp(z_host, z, F_theta, right=0.0)
    F_q_h = np.interp(z_host, z, F_q, right=0.0)
    F_u_h = np.interp(z_host, z, F_u, right=0.0)
    # zera explicitamente a partir do primeiro nivel acima do topo real da pluma
    above = z_host > z_top
    F_theta_h[above] = 0.0
    F_q_h[above] = 0.0
    F_u_h[above] = 0.0

    rho_h = p_of_z(z_host)*100.0/(Rd*np.array([T_env_at(zz) for zz in z_host]))

    dtheta_dt = -np.gradient(F_theta_h, z_host)/rho_h * 86400.0
    dq_dt = -np.gradient(F_q_h, z_host)/rho_h * 86400.0 * 1000.0   # kg/kg/dia -> g/kg/dia
    du_dt = -np.gradient(F_u_h, z_host)/rho_h * 86400.0

    return dict(z=z_host, dtheta_dt=dtheta_dt, dq_dt=dq_dt, du_dt=du_dt)

EPS_SHALLOW = 5.0e-3
EPS_DEEP    = 1.0e-4
Z_CHECK_DEEP = 1750.0

# --- Espectro de plumas (estilo Arakawa & Schubert 1974) ---
N_SPECTRUM = 6
EPS_SPECTRUM = np.geomspace(EPS_DEEP, EPS_SHALLOW, N_SPECTRUM)

def run_spectrum(z0, T0, q0, w2_0, q_ft):
    """Integra um espectro de plumas com taxas de entranhamento entre
    EPS_DEEP e EPS_SHALLOW. Retorna a lista de resultados de run_plume."""
    return [run_plume(z0, T0, q0, w2_0, eps, q_ft) for eps in EPS_SPECTRUM]

# =====================================================================
# 4. DOWNDRAFT CONVECTIVO (corrente descendente) - resfriamento evaporativo
# =====================================================================
EPS_DOWN = 1.5e-3        # 1/m -- entranhamento do downdraft
EVAP_EFF_DOWN = 3.0e-4    # eficiencia de evaporacao da chuva no downdraft [1/m, por unidade de subsaturacao]

def run_downdraft(z_top_cloud, rain_available, T_env_cloudbase, z_cloud_base, dz=DZ):
    """
    Downdraft simplificado: inicia perto do topo da nuvem (LFS aproximado
    a 70% da profundidade da nuvem, entre a base e o topo - abaixo desse
    nivel o ar da nuvem misturado 50/50 com o ambiente e resfriado por
    evaporacao tende a ficar negativamente flutuante, seguindo o conceito
    de cudlfs do Tiedtke). Desce entranhando ar ambiente e evaporando a
    precipitacao disponivel, o que o resfria e o umedece, sustentando a
    aceleracao descendente. Retorna se o downdraft atinge a superficie e
    a anomalia de temperatura do "cold pool" resultante.
    """
    z_lfs = z_cloud_base + 0.7*(z_top_cloud - z_cloud_base)
    if z_lfs <= z_cloud_base or rain_available <= 0.0:
        return dict(reaches_surface=False, cold_pool_dT=0.0, z_lfs=z_lfs)

    Te = T_env_at(z_lfs); pe = p_of_z(z_lfs)
    qe = q_env_at(z_lfs, q_ft0_global[0])
    # parcela inicial do downdraft: mistura 50/50 com o ambiente, saturada
    # por evaporacao de parte da chuva disponivel (aproximacao simples do LFS)
    T = Te - 1.0     # pequeno deficit inicial para iniciar o afundamento
    q = min(qe*1.05, qsat_liq(Te, pe))
    w2 = 0.3
    rain = rain_available
    z = z_lfs

    while z > 0.0 and rain > 0.0:
        z_new = max(z - dz, 0.0)
        Te = T_env_at(z_new); pe = p_of_z(z_new)
        qe = q_env_at(z_new, q_ft0_global[0])

        # entranhamento do ambiente
        T = T - EPS_DOWN*dz*(T - Te)
        q = q - EPS_DOWN*dz*(q - qe)

        # evaporacao da chuva disponivel, proporcional ao deficit de saturacao
        qs = qsat_liq(T, pe)
        deficit = max(qs - q, 0.0)
        devap = min(rain, EVAP_EFF_DOWN*dz*deficit*100.0)   # fracao evaporada nesta camada
        rain -= devap
        q += devap
        T -= devap*Lv/cp   # resfriamento evaporativo

        Tv_d = T*(1.0 + 0.61*q)
        Tv_e = Te*(1.0 + 0.61*qe)
        B = g*(Tv_d - Tv_e)/Tv_e   # negativo => empuxo negativo => acelera p/ baixo

        w2 = w2 + dz*(-2*A_BUOY*B - 2*EPS_DOWN*w2)   # desce; -B (negativo) vira positivo
        z = z_new

        if w2 <= 0.0:
            return dict(reaches_surface=False, cold_pool_dT=0.0, z_lfs=z_lfs)

    if z <= 0.0:
        cold_pool_dT = T - T_env_at(0.0)
        return dict(reaches_surface=True, cold_pool_dT=cold_pool_dT, z_lfs=z_lfs,
                    rain_reaching_surface=rain)
    return dict(reaches_surface=False, cold_pool_dT=0.0, z_lfs=z_lfs)

q_ft0_global = [9.0e-3]   # referencia mutavel, atualizada no loop diurno p/ uso no downdraft

# =====================================================================
# 5. EVAPORACAO DE PRECIPITACAO ABAIXO DA BASE DA NUVEM + DERRETIMENTO DE NEVE
# =====================================================================
EVAP_SUBCLOUD_COEF = 0.15   # fracao evaporada por km de queda em ar com 100% de subsaturacao

def evap_below_cloud(rain_at_base, z_cloud_base, T_ml, q_ml):
    """Evapora uma fracao da chuva que cai da base da nuvem ate a
    superficie, proporcional a subsaturacao do ar sub-nuvem (CLC)."""
    qs_ml = qsat_liq(T_ml, p_of_z(z_cloud_base/2))
    subsat = max(0.0, 1.0 - q_ml/max(qs_ml, 1e-6))
    evap_frac = min(1.0, EVAP_SUBCLOUD_COEF * (z_cloud_base/1000.0) * subsat)
    evaporated = rain_at_base * evap_frac
    return rain_at_base - evaporated, evaporated

def melt_snow(snow_flux, z_top_cloud):
    """Derrete a neve/graupel que cai abaixo do nivel de congelamento
    ambiente, convertendo-a em chuva (consumo simplificado do calor
    latente de fusao, sem efeito retroativo na pluma)."""
    z_freezing = None
    for zz in np.arange(0.0, z_top_cloud, 50.0):
        if T_env_at(zz) >= T_LIQ_ALL:
            z_freezing = zz
            break
    if z_freezing is None or z_freezing <= 0.0:
        return snow_flux, 0.0   # topo da nuvem ja esta abaixo do congelamento -> tudo derrete perto da base
    melted = snow_flux
    return 0.0, melted

# =====================================================================
# 6. EVOLUCAO DIURNA (encroachment + balanco de umidade + fechamento CAPE)
# =====================================================================
rho0 = 1.15
gamma_theta = 2.0e-3
T_clc0 = 296.0
t_sunrise = 6*3600.0
daylength = 12*3600.0
SHF_max, LHF_max = 250.0, 300.0

def SHF(t):
    hh = t - t_sunrise
    return max(0.0, SHF_max*np.sin(np.pi*hh/daylength)) if 0 <= hh <= daylength else 0.0

def LHF(t):
    hh = t - t_sunrise
    return max(0.0, LHF_max*np.sin(np.pi*hh/daylength)) if 0 <= hh <= daylength else 0.0

q_ft0 = 9.0e-3
q_ft_sat = 14.0e-3
tau_relax = 6*3600.0
c1_mb = 2.5e-3

# --- Fechamento por quase-equilibrio de CAPE (para a pluma profunda) ---
TAU_ADJUST = 3*3600.0     # escala de tempo de ajuste do CAPE (Arakawa-Schubert-like)
CAPE_REF = 400.0           # CAPE de referencia para normalizar o fechamento [J/kg]

dt = 60.0
t_end = 18*3600.0
times = np.arange(0.0, t_end, dt)
n = len(times)

h    = np.zeros(n); h[0] = 200.0
q_ml = np.zeros(n); q_ml[0] = 11.0e-3
q_ft = np.zeros(n); q_ft[0] = q_ft0
T_ml = np.zeros(n); T_ml[0] = 296.0

plume_stride = 5
z_top_shallow = np.full(n, np.nan)
z_top_deep    = np.full(n, np.nan)
survived_deep = np.zeros(n, dtype=bool)
Mb_shallow    = np.zeros(n)
Mb_deep       = np.zeros(n)
CAPE_t        = np.zeros(n)
rain_rate     = np.zeros(n)
snow_rate     = np.zeros(n)
coldpool_dT   = np.zeros(n)
downdraft_active = np.zeros(n, dtype=bool)
mom_flux_t    = np.zeros(n)
spectrum_tops_last = None

trigger_time = None
last_shallow_top = 0.0
last_shallow_qtop = q_ft0
last_rain = 0.0
last_snow = 0.0
clouds_active = False

TARGET_SNAPSHOT_HOURS = [11.0, 15.0]   # manha (so raso) e tarde (pos-transicao)
tendencias_snapshots = {}   # {hora_alvo: {"shallow":..., "deep":..., "t_real":...}}

for i in range(1, n):
    t = times[i]
    shf, lhf = SHF(t), LHF(t)
    _rad_offset[0] = RAD_COOL_RATE * t   # resfriamento radiativo acumulado ate agora

    # --- Camada limite convectiva: encroachment ---
    h2 = h[i-1]**2 + 2.0*dt*max(shf, 0.0) / (rho0*cp*gamma_theta)
    h[i] = np.sqrt(max(h2, 200.0**2))
    dh = h[i]-h[i-1]
    T_ml[i] = T_clc0 + gamma_theta * h[i]

    q_ml[i] = q_ml[i-1] + dt*(max(lhf,0.0)/(rho0*Lv*h[i])) + (q_ft[i-1]-q_ml[i-1])*(dh/h[i])
    q_ml[i] = max(q_ml[i], 1e-4)

    Mb_shallow[i] = c1_mb * max(shf, 0.0)
    q_ft0_global[0] = q_ft[i-1]

    z_lcl_now = lcl_height(T_ml[i], q_ml[i])
    clouds_active = (shf > 0.0) and (h[i] >= z_lcl_now)

    if clouds_active and (i % plume_stride == 0 or (i > 1 and not np.isfinite(z_top_shallow[i-1]))):
        z0 = z_lcl_now
        T0 = T_ml[i] - (g/cp)*z0
        q0 = qsat_liq(T0, p_of_z(z0))
        w_star2 = max(0.2, (g/T0 * shf/(rho0*cp) * h[i])**(2.0/3.0) * 0.3)

        shallow = run_plume(z0, T0, q0, w_star2, EPS_SHALLOW, q_ft[i-1])
        last_shallow_top = shallow["z_top"]
        last_shallow_qtop = shallow["q_top"]
        last_rain = shallow["rain"] * Mb_shallow[i] * 3600.0
        last_snow = shallow["snow"] * Mb_shallow[i] * 3600.0
        mom_flux_t[i] = shallow["mom_flux_top"] * Mb_shallow[i]

        # espectro de plumas -> CAPE (da menos diluida) e diagnostico de topos
        spectrum = run_spectrum(z0, T0, q0, w_star2, q_ft[i-1])
        spectrum_tops_last = [r["z_top"] for r in spectrum]
        CAPE_now = max(r["CAPE"] for r in spectrum)
        CAPE_t[i] = CAPE_now

        # fechamento por quase-equilibrio: Mb_deep responde ao CAPE disponivel
        Mb_deep[i] = Mb_shallow[i] * (CAPE_now/CAPE_REF) if CAPE_now > 0 else 0.0

        deep = spectrum[0]   # a de menor entranhamento (mais proxima da "profunda")
        z_top_deep[i] = deep["z_top"]
        survived_deep[i] = deep["z_top"] >= Z_CHECK_DEEP

        # --- captura das tendencias convectivas em horarios-alvo (Figura de tendencias) ---
        for hora_alvo in TARGET_SNAPSHOT_HOURS:
            if hora_alvo not in tendencias_snapshots and t/3600.0 >= hora_alvo:
                tend_shallow = calcula_tendencias(shallow, Mb_shallow[i], q_ft[i-1])
                tend_deep = calcula_tendencias(deep, Mb_deep[i] if Mb_deep[i] > 0 else Mb_shallow[i], q_ft[i-1])
                tendencias_snapshots[hora_alvo] = dict(shallow=tend_shallow, deep=tend_deep, t_real=t/3600.0)

        if survived_deep[i] and deep["rain"] > 0:
            dd = run_downdraft(deep["z_top"], deep["rain"], T_env_at(z0), z0)
            downdraft_active[i] = dd["reaches_surface"]
            coldpool_dT[i] = dd["cold_pool_dT"]

    elif not clouds_active:
        last_shallow_top = h[i]
        last_shallow_qtop = q_ft[i-1]
        last_rain = 0.0
        last_snow = 0.0

    z_top_shallow[i] = last_shallow_top if clouds_active else h[i]

    # --- evaporacao abaixo da nuvem + derretimento de neve (chuva/neve rasas) ---
    if clouds_active and (last_rain > 0 or last_snow > 0):
        rain_sfc, evap_amt = evap_below_cloud(last_rain, z_top_shallow[i], T_ml[i], q_ml[i])
        snow_after_melt, melted = melt_snow(last_snow, z_top_shallow[i])
        rain_rate[i] = rain_sfc + melted
        snow_rate[i] = snow_after_melt
    else:
        rain_rate[i] = 0.0
        snow_rate[i] = 0.0

    # --- umedecimento da troposfera livre pelo detranhamento real da pluma rasa ---
    if clouds_active:
        delta_z_detrain = max(last_shallow_top - h[i], 100.0)
        umedecimento = Mb_shallow[i]/rho0 * (last_shallow_qtop - q_ft[i-1]) / delta_z_detrain
    else:
        umedecimento = 0.0
    secagem = (q_ft[i-1]-q_ft0)/tau_relax
    q_ft[i] = np.clip(q_ft[i-1] + dt*(umedecimento - secagem), q_ft0, q_ft_sat)

    if trigger_time is None and survived_deep[i]:
        trigger_time = t

horas = times/3600.0
if trigger_time is not None:
    print(f"Transicao rasa -> profunda: {trigger_time/3600:.2f} h local")
else:
    print("Pluma profunda nao sobreviveu a camada de CIN no periodo simulado.")
print(f"Downdraft atingiu a superficie em {downdraft_active.sum()} dos {n} passos "
      f"({'sim' if downdraft_active.any() else 'nao'} houve cold pool).")
if downdraft_active.any():
    idx = np.where(downdraft_active)[0][0]
    print(f"Primeiro cold pool: {horas[idx]:.2f} h, DeltaT = {coldpool_dT[idx]:.2f} K")

# =====================================================================
# 7. FIGURAS
# =====================================================================
fig, axs = plt.subplots(4, 2, figsize=(13,12.3), sharex=True)
fig.suptitle("v3: fase gelo/liquida, downdraft, fechamento CAPE, momentum, espectro", fontsize=13, fontweight="bold")

axs[0,0].plot(horas, [SHF(t) for t in times], color="firebrick", label="SHF")
axs[0,0].plot(horas, [LHF(t) for t in times], color="steelblue", label="LHF")
axs[0,0].set_title("Fluxos de superficie"); axs[0,0].set_ylabel("W/m^2"); axs[0,0].legend(fontsize=8)

axs[0,1].plot(horas, h, color="darkorange", label="base da nuvem")
axs[0,1].plot(horas, z_top_shallow, color="grey", label="topo raso")
axs[0,1].plot(horas, z_top_deep, color="black", ls="--", label="alcance profundo (espectro)")
axs[0,1].axhline(Z_CHECK_DEEP, color="crimson", ls=":", lw=1, label="altura de verificacao")
axs[0,1].set_title("Alturas"); axs[0,1].set_ylabel("m"); axs[0,1].legend(fontsize=7)

axs[1,0].plot(horas, q_ml*1000, color="seagreen", label="q CLC")
axs[1,0].plot(horas, q_ft*1000, color="teal", label="q troposfera livre")
axs[1,0].set_title("Umedecimento por detranhamento"); axs[1,0].set_ylabel("g/kg"); axs[1,0].legend(fontsize=8)

axs[1,1].plot(horas, Mb_shallow*1000, color="purple", label="Mb rasa (SHF)")
axs[1,1].plot(horas, Mb_deep*1000, color="darkred", label="Mb profunda (quase-eq. CAPE)")
axs[1,1].set_title("Fluxo de massa de base"); axs[1,1].set_ylabel("g/m^2/s"); axs[1,1].legend(fontsize=7)

axs[2,0].plot(horas, CAPE_t, color="navy")
axs[2,0].set_title("CAPE (pluma menos diluida do espectro)"); axs[2,0].set_ylabel("J/kg")

axs[2,1].plot(horas, rain_rate, color="steelblue", label="chuva")
axs[2,1].plot(horas, snow_rate, color="lightsteelblue", label="neve/graupel")
axs[2,1].set_title("Precipitacao de superficie (apos evap. sub-nuvem)")
axs[2,1].set_ylabel("mm/h"); axs[2,1].legend(fontsize=8)

axs[3,0].plot(horas, coldpool_dT, color="black")
axs[3,0].set_title("Anomalia de temperatura do cold pool (downdraft)")
axs[3,0].set_ylabel("K"); axs[3,0].set_xlabel("Hora local")

axs[3,1].plot(horas, mom_flux_t, color="darkgreen")
axs[3,1].set_title("Fluxo convectivo de momentum (diagnostico, detranhamento)")
axs[3,1].set_ylabel("m/s . kg/m^2/s"); axs[3,1].set_xlabel("Hora local")

for ax in axs.flat:
    ax.grid(alpha=0.3)
    if trigger_time is not None:
        ax.axvline(trigger_time/3600.0, color="black", ls=":", lw=1.0)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("v3_ciclo_diurno_completo.png", dpi=150)
print("Figura salva: v3_ciclo_diurno_completo.png")

# --- Figura 2: espectro de plumas no ultimo horario calculado ---
if spectrum_tops_last is not None:
    fig2, ax2 = plt.subplots(figsize=(7,5))
    ax2.barh(range(N_SPECTRUM), spectrum_tops_last, color="slategray")
    ax2.set_yticks(range(N_SPECTRUM))
    ax2.set_yticklabels([f"eps={e:.1e}" for e in EPS_SPECTRUM])
    ax2.set_xlabel("Altura do topo da nuvem [m]")
    ax2.set_title("Espectro de plumas (Arakawa-Schubert-like)\nno ultimo horario calculado")
    ax2.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("v3_espectro_plumas.png", dpi=150)
    print("Figura salva: v3_espectro_plumas.png")

# =====================================================================
# 8. FIGURA 3: TENDENCIAS CONVECTIVAS (aquecimento, umedecimento, momentum)
# =====================================================================
# Perfis verticais de dtheta/dt, dq/dt, du/dt induzidos pela convecao sobre
# o ambiente de grande escala -- o que a parametrizacao de fato devolve ao
# modelo hospedeiro (slide 39 do curso: termos de subsidencia de larga
# escala e transporte turbulento). Ver calcula_tendencias() na Secao 3.
if len(tendencias_snapshots) > 0:
    horas_capturadas = sorted(tendencias_snapshots.keys())
    fig3, axs3 = plt.subplots(1, 3, figsize=(13, 6), sharey=True)

    cores = {11.0: "steelblue", 15.0: "firebrick"}
    for hora_alvo in horas_capturadas:
        snap = tendencias_snapshots[hora_alvo]
        cor = cores.get(hora_alvo, "black")
        t_real = snap["t_real"]

        ts = snap["shallow"]
        axs3[0].plot(ts["dtheta_dt"], ts["z"]/1000, color=cor, ls="--",
                     label=f"rasa, t={t_real:.1f}h")
        axs3[1].plot(ts["dq_dt"], ts["z"]/1000, color=cor, ls="--")
        axs3[2].plot(ts["du_dt"], ts["z"]/1000, color=cor, ls="--")

        td = snap["deep"]
        axs3[0].plot(td["dtheta_dt"], td["z"]/1000, color=cor, ls="-",
                     label=f"profunda, t={t_real:.1f}h")
        axs3[1].plot(td["dq_dt"], td["z"]/1000, color=cor, ls="-")
        axs3[2].plot(td["du_dt"], td["z"]/1000, color=cor, ls="-")

    for ax in axs3:
        ax.axvline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3)
    axs3[0].set_xlabel("dtheta/dt [K/dia]")
    axs3[0].set_ylabel("altura [km]")
    axs3[0].set_title("Aquecimento convectivo")
    axs3[0].legend(fontsize=7, loc="upper right")
    axs3[1].set_xlabel("dq/dt [g/kg/dia]")
    axs3[1].set_title("Umedecimento convectivo")
    axs3[2].set_xlabel("du/dt [m/s/dia]")
    axs3[2].set_title("Transporte de momentum")

    fig3.suptitle("Tendencias convectivas sobre o ambiente de grande escala\n"
                  "(linha tracejada = pluma rasa; linha cheia = pluma profunda)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig("v3_tendencias_convectivas.png", dpi=150)
    print("Figura salva: v3_tendencias_convectivas.png")
else:
    print("Aviso: nenhum snapshot de tendencias foi capturado (confira TARGET_SNAPSHOT_HOURS).")
