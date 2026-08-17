# -*- coding: utf-8 -*-
"""
MODELO CONCEITUAL DA TRANSICAO RASA -> PROFUNDA - v2
======================================================
Curso: Conveccao Atmosferica - Regimes de Transicao

DIFERENCA EM RELACAO A v1
--------------------------
Na v1, o CIN e o fluxo de massa eram estimados por formulas heuristicas
escalares (CIN(t) = CIN0*exp(...), Mb = c1*SHF). Isso deixava de fora o
processo fisico central de qualquer parametrizacao de cumulus: o FLUXO
DE MASSA propriamente dito, isto e, a integracao vertical da pluma
convectiva com entranhamento/detranhamento (de Rooy et al. 2013;
Arakawa & Schubert 1974; Simpson & Wiggert 1969).

Esta versao substitui essas formulas por um MODELO DE PLUMA ENTRANHANTE
explicito, integrado na vertical a cada passo de tempo, para DOIS
regimes de entranhamento (raso e profundo). O CIN, o CAPE, a altura do
topo da nuvem rasa e o instante de disparo da conveccao profunda
deixam de ser impostos e passam a EMERGIR da fisica de empuxo +
entranhamento.

PROCESSOS FISICOS REPRESENTADOS
--------------------------------
  (a) Crescimento da CLC por aquecimento de superficie (encroachment);
  (b) Balanco de umidade da CLC;
  (c) Sondagem ambiente com estrutura CIN (camada estavel rasa) + CAPE
      (camada condicionalmente instavel profunda), construida a partir
      de um perfil de parcela nao-diluida;
  (d) PLUMA ENTRANHANTE RASA: integracao vertical de dtheta/dz, dq/dz
      (mistura turbulenta com o ambiente) e d(w^2)/dz (empuxo vs. arrasto
      por entranhamento) partindo da base da nuvem - define a altura do
      topo raso onde a pluma perde flutuabilidade e DETRANHA;
  (e) Umedecimento da troposfera livre pelo detranhamento da pluma rasa
      (no nivel fisico onde ela de fato para, com a umidade que ela de
      fato carrega - nao mais um valor escalar arbitrario);
  (f) PLUMA ENTRANHANTE PROFUNDA (teste): mesma fisica, com taxa de
      entranhamento muito menor (nuvem mais larga). A cada passo de
      tempo, testamos se essa pluma sobrevive (w^2>0) atraves da camada
      de CIN. A transicao rasa->profunda e o instante em que ela passa
      a sobreviver - o "gatilho" emerge da diluicao por entranhamento
      ficar pequena o suficiente, e nao de um limiar prescrito.

Ainda e um modelo DIDATICO (fechamentos simplificados, sondagem
ambiente idealizada), mas agora o fluxo de massa e calculado de
verdade, camada por camada, com as equacoes de pluma.
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
Lv = 2.5e6
eps_R = Rd / Rv  # 0.622

def qsat(T, p):
    """Umidade especifica de saturacao [kg/kg]. T em K, p em hPa (Bolton 1980)."""
    Tc = T - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))   # hPa
    return eps_R * es / np.maximum(p - es, 1e-3)

def moist_lapse(T, p):
    """Taxa de resfriamento adiabatico saturado (K/m), formulacao padrao."""
    rs = qsat(T, p)
    num = g * (1.0 + Lv * rs / (Rd * T))
    den = cp + (Lv**2 * rs * eps_R) / (Rd * T**2)
    return num / den

def p_of_z(z, p0=1000.0, H=8000.0):
    """Pressao hidrostatica aproximada [hPa] (escala fixa, simplificacao didatica)."""
    return p0 * np.exp(-z / H)

# =====================================================================
# 2. SONDAGEM AMBIENTE (fixa, construida para gerar CIN raso + CAPE profundo)
# =====================================================================
# =====================================================================
# 2. SONDAGEM AMBIENTE (prescrita por taxas de lapso em camadas)
# =====================================================================
Z_TOP = 14000.0
DZ = 20.0   # resolucao vertical fina - necessaria para resolver bem o ponto onde w^2->0
zgrid = np.arange(0.0, Z_TOP + DZ, DZ)
pgrid = p_of_z(zgrid)

T_env0 = 297.0   # temperatura ambiente de superficie [K] (referencia fixa, independente da CLC)

# Camadas (m) e taxas de lapso (K/m): quase-adiabatica seca -> ESTAVEL (gera CIN) ->
# condicionalmente instavel (gera CAPE) -> isotermica (tropopausa/EL)
LAPSE_SEGMENTS = [(0.0, 1400.0, 9.0e-3),
                  (1400.0, 1950.0, 2.8e-3),
                  (1950.0, 11000.0, 7.0e-3),
                  (11000.0, 20000.0, 0.0)]

def lapse_rate(z):
    for zlo, zhi, rate in LAPSE_SEGMENTS:
        if zlo <= z < zhi:
            return rate
    return LAPSE_SEGMENTS[-1][2]

T_env = np.zeros_like(zgrid)
T_env[0] = T_env0
for i in range(1, len(zgrid)):
    T_env[i] = T_env[i-1] - lapse_rate(zgrid[i-1]) * (zgrid[i]-zgrid[i-1])

z_transition_top = 3000.0   # ate aqui, a umidade ambiente e controlada por q_ft(t) (umedecimento raso)
RH_bg = 0.40                 # umidade relativa de fundo acima da camada de transicao

def T_env_at(z):
    return np.interp(z, zgrid, T_env)

def q_env_at(z, q_ft):
    """Umidade ambiente: camada de transicao -> q_ft(t) ; acima -> RH de fundo fixa."""
    if z <= z_transition_top:
        return q_ft
    Te = T_env_at(z)
    return qsat(Te, p_of_z(z)) * RH_bg

def lcl_height(T_k, q):
    """Aproximacao classica do LCL (Espy): z_LCL[m] ~= 125*(T-Td)[degC]."""
    Tc = T_k - 273.15
    es = q * 1000.0 / (eps_R + q*(1-eps_R))
    es = max(es, 1e-6)
    Td = 243.5*np.log(es/6.112) / (17.67 - np.log(es/6.112))
    return max(0.0, 125.0*(Tc - Td))

# =====================================================================
# 3. MODELO DE PLUMA ENTRANHANTE (fluxo de massa) - mistura termodinamicamente
#    consistente: entranhamento mistura energia estatica liquida (sl) e agua
#    total (qt), NAO temperatura/umidade diretamente. Isso captura o
#    RESFRIAMENTO EVAPORATIVO do ar seco entranhado ao evaporar agua de nuvem
#    para saturar a mistura - o mecanismo fisico real que liga o umedecimento
#    ambiente a reducao da diluicao por entranhamento (de Rooy et al. 2013).
# =====================================================================
A_BUOY = 1.0   # fracao do empuxo convertida em aceleracao vertical
B_DRAG = 1.0   # coeficiente de arrasto por entranhamento

# --- Formacao de precipitacao (autoconversao agua de nuvem -> chuva) ---
# Segue diretamente a formulacao operacional do esquema de Tiedtke (rotina
# cuasc do module_cu_tiedtke_F.txt): a cada nivel, uma fracao da agua liquida
# da pluma e convertida em precipitacao, mas somente depois que a nuvem tem
# profundidade suficiente acima da base (senao as goticulas ainda nao
# cresceram o bastante para precipitar):
#
#   zlnew = plu / (1 + cprcon * g * dz)      [cuasc, module_cu_tiedtke_F.txt]
#   pdmfup = max(0, (plu - zlnew) * mfu)      <- fluxo de precipitacao gerado
#
# cprcon = 1.1e-3/g  =>  em termos de dz [m], o coeficiente efetivo e 1.1e-3/m.
CPRCON_RATE = 1.1e-3     # 1/m -- taxa de autoconversao (Tiedtke: cprcon*g)
DEPTH_NO_PRECIP = 100.0  # m acima da base da nuvem (reescalado para a profundidade de nuvem
                           # deste modelo simplificado; no Tiedtke real, zdnoprc=1.5e4 Pa ~1200-1500m,
                           # mas nossas plumas sao muito mais rasas em termos absolutos - ver nota no texto)

def sat_adjust(sl, qt, z, p):
    """Resolve (T, q_vapor, q_liquido) dado sl e qt, por iteracao de ponto fixo."""
    T = (sl - g*z) / cp
    for _ in range(8):
        qs = qsat(T, p)
        ql = max(0.0, qt - qs)
        T_new = (sl - g*z + Lv*ql) / cp
        if abs(T_new - T) < 1e-4:
            T = T_new
            break
        T = T_new
    qs = qsat(T, p)
    ql = max(0.0, qt - qs)
    return T, qt - ql, ql

def run_plume(z0, T0, q0, w2_0, epsilon, q_ft, zmax=Z_TOP, dz=DZ):
    """
    Integra uma pluma entranhante a partir de (z0, T0, q0), saturada, com
    taxa de entranhamento fracionario 'epsilon' [1/m]. A cada nivel, aplica
    autoconversao de agua de nuvem em precipitacao (formulacao Tiedtke).
    Retorna a altura onde a pluma detranha (w^2 <= 0), a umidade que ela
    carrega nesse nivel, e a taxa de precipitacao de superficie equivalente.
    """
    n = int((zmax - z0) / dz) + 1
    z = z0 + np.arange(n) * dz
    B_arr = np.zeros(n)
    sl = cp*T0 + g*z0          # energia estatica liquida (ql0=0, recem-saturada)
    qt = q0
    w2 = w2_0
    top_idx = n - 1
    q_final = q0
    precip_flux = 0.0           # fluxo de precipitacao acumulado [kg/kg * (kg/m2/s equiv.)]

    for i in range(1, n):
        Te = T_env_at(z[i]); pe = p_of_z(z[i])
        qe = q_env_at(z[i], q_ft)
        sl_env = cp*Te + g*z[i]
        qt_env = qe

        # entranhamento: mistura sl e qt (conservados) com o ambiente
        sl = sl - epsilon*dz*(sl - sl_env)
        qt = qt - epsilon*dz*(qt - qt_env)

        T, q, ql = sat_adjust(sl, qt, z[i], pe)

        # --- autoconversao (formacao de precipitacao), formulacao Tiedtke ---
        if (z[i] - z0) >= DEPTH_NO_PRECIP and ql > 0.0:
            ql_new = ql / (1.0 + CPRCON_RATE*dz)
            dprecip = ql - ql_new     # agua removida da pluma nesta camada
            precip_flux += dprecip     # acumula ao longo da coluna [kg/kg equivalente]
            ql = ql_new
            qt = q + ql                # atualiza agua total (removendo o que precipitou)
            sl = cp*T + g*z[i] - Lv*ql  # recalcula sl consistente com o novo ql

        Tv_u = T*(1.0 + 0.61*q - ql)
        Tv_e = Te*(1.0 + 0.61*qe)
        B = g*(Tv_u - Tv_e)/Tv_e
        B_arr[i] = B

        w2 = w2 + dz*(2*A_BUOY*B - 2*B_DRAG*epsilon*w2)
        q_final = q

        if w2 <= 0.0:
            top_idx = i
            break

    return dict(z_top=z[top_idx], q_top=q_final, w2_end=max(w2, 0.0),
                z=z[:top_idx+1], B=B_arr[:top_idx+1], precip=precip_flux)

# Taxas de entranhamento tipicas (de Rooy et al. 2013): rasa >> profunda
EPS_SHALLOW = 5.0e-3   # 1/m -> nuvens estreitas, muito diluidas
EPS_DEEP    = 1.0e-4   # 1/m -> nuvens largas, pouco diluidas
Z_CHECK_DEEP = 1750.0  # altura de verificacao: se a pluma profunda sobrevive ate aqui, disparo ocorreu

# =====================================================================
# 4. EVOLUCAO DIURNA DA CAMADA LIMITE (encroachment + balanco de umidade)
# =====================================================================
rho0 = 1.15
gamma_theta = 2.0e-3
T_clc0 = 296.0          # temperatura de superficie da CLC ao amanhecer [K] (ligeiramente mais fria que o ambiente, resfriamento noturno)
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
c1_mb = 2.5e-3          # eficiencia SHF -> fluxo de massa de base de nuvem rasa

dt = 60.0
t_end = 18*3600.0
times = np.arange(0.0, t_end, dt)
n = len(times)

h    = np.zeros(n); h[0] = 200.0
q_ml = np.zeros(n); q_ml[0] = 11.0e-3
q_ft = np.zeros(n); q_ft[0] = q_ft0
T_ml = np.zeros(n); T_ml[0] = 296.0

# saidas diagnosticas (recalculadas em cadencia reduzida p/ custo computacional)
plume_stride = 5   # roda o modelo de pluma a cada 5 passos (5 min)
z_top_shallow = np.full(n, np.nan)
z_top_deep    = np.full(n, np.nan)
survived_deep = np.zeros(n, dtype=bool)
Mb_shallow    = np.zeros(n)
precip_shallow_rate = np.zeros(n)   # taxa de precipitacao da pluma rasa [mm/h equivalente]
precip_deep_rate    = np.zeros(n)   # taxa de precipitacao da pluma profunda (quando ativa) [mm/h]

def lcl_height(T_k, q):
    """Aproximacao classica do LCL (Espy): z_LCL[m] ~= 125*(T-Td)[degC]."""
    Tc = T_k - 273.15
    p_here = 1000.0
    es = q * p_here / (eps_R + q*(1-eps_R))
    es = max(es, 1e-6)
    Td = 243.5*np.log(es/6.112) / (17.67 - np.log(es/6.112))
    return max(0.0, 125.0*(Tc - Td))

trigger_time = None
last_shallow_top = 0.0
last_shallow_qtop = q_ft0
last_precip_shallow = 0.0
last_precip_deep = 0.0
clouds_active = False

for i in range(1, n):
    t = times[i]
    shf, lhf = SHF(t), LHF(t)

    # --- Camada limite convectiva: encroachment ---
    h2 = h[i-1]**2 + 2.0*dt*max(shf, 0.0) / (rho0*cp*gamma_theta)
    h[i] = np.sqrt(max(h2, 200.0**2))
    dh = h[i]-h[i-1]

    # Temperatura potencial da CLC: diagnosticada pelo proprio encroachment
    # (theta_CLC = theta_env no topo da CLC, por definicao do modelo de encroachment).
    # Isso evita uma inconsistencia termodinamica entre o crescimento de h(t)
    # e uma integracao independente de temperatura.
    T_ml[i] = T_clc0 + gamma_theta * h[i]

    q_ml[i] = q_ml[i-1] + dt*(max(lhf,0.0)/(rho0*Lv*h[i])) + (q_ft[i-1]-q_ml[i-1])*(dh/h[i])
    q_ml[i] = max(q_ml[i], 1e-4)

    Mb_shallow[i] = c1_mb * max(shf, 0.0)

    # --- Conveccao rasa so existe depois que a CLC atinge seu proprio LCL, com sol no ceu ---
    z_lcl_now = lcl_height(T_ml[i], q_ml[i])
    clouds_active = (shf > 0.0) and (h[i] >= z_lcl_now)

    if clouds_active and (i % plume_stride == 0 or (i > 1 and not np.isfinite(z_top_shallow[i-1]))):
        z0 = z_lcl_now   # base da nuvem = LCL real (nao h[i], que pode ultrapassar o LCL)
        T0 = T_ml[i] - (g/cp)*z0                     # extrapola theta_CLC (T_ml) ate z0 pelo adiabatico seco
        q0 = qsat(T0, p_of_z(z0))                     # saturada no LCL
        w_star2 = max(0.2, (g/T0 * shf/(rho0*cp) * h[i])**(2.0/3.0) * 0.3)

        shallow = run_plume(z0, T0, q0, w_star2, EPS_SHALLOW, q_ft[i-1])
        last_shallow_top = shallow["z_top"]
        last_shallow_qtop = shallow["q_top"]
        # taxa de precipitacao de superficie: fluxo de massa de base x agua convertida em chuva
        # (kg/kg * kg/m2/s = kg/m2/s de agua; x3600 -> mm/h, ja que 1 kg/m2 de agua = 1 mm)
        last_precip_shallow = shallow["precip"] * Mb_shallow[i] * 3600.0

        deep = run_plume(z0, T0, q0, w_star2, EPS_DEEP, q_ft[i-1])
        z_top_deep[i] = deep["z_top"]
        survived_deep[i] = deep["z_top"] >= Z_CHECK_DEEP
        # usa o mesmo fluxo de massa de base como referencia ilustrativa
        # (o modelo nao tem um fechamento de fluxo de massa profundo proprio)
        last_precip_deep = deep["precip"] * Mb_shallow[i] * 3600.0 if survived_deep[i] else 0.0
    elif not clouds_active:
        last_shallow_top = h[i]
        last_shallow_qtop = q_ft[i-1]
        last_precip_shallow = 0.0
        last_precip_deep = 0.0

    precip_shallow_rate[i] = last_precip_shallow if clouds_active else 0.0
    precip_deep_rate[i] = last_precip_deep if clouds_active else 0.0

    z_top_shallow[i] = last_shallow_top if clouds_active else h[i]

    # --- Umedecimento da troposfera livre pelo detranhamento REAL da pluma rasa ---
    if clouds_active:
        delta_z_detrain = max(last_shallow_top - h[i], 100.0)
        umedecimento = Mb_shallow[i]/rho0 * (last_shallow_qtop - q_ft[i-1]) / delta_z_detrain
    else:
        umedecimento = 0.0
    secagem = (q_ft[i-1]-q_ft0)/tau_relax
    q_ft[i] = np.clip(q_ft[i-1] + dt*(umedecimento - secagem), q_ft0, q_ft_sat)

    if trigger_time is None and survived_deep[i]:
        trigger_time = t

# preencher NaNs de z_top_deep para plotagem (mantem ultimo valor valido)
for arr in [z_top_deep]:
    last = np.nan
    for i in range(n):
        if np.isnan(arr[i]):
            arr[i] = last
        else:
            last = arr[i]

horas = times/3600.0
if trigger_time is not None:
    print(f"Transicao rasa -> profunda (pluma profunda sobrevive a camada de CIN): {trigger_time/3600:.2f} h local")
else:
    print("Pluma profunda nao sobreviveu a camada de CIN no periodo simulado.")

# =====================================================================
# 5. FIGURAS
# =====================================================================
fig, axs = plt.subplots(4, 2, figsize=(13,12.3), sharex=True)
fig.suptitle("Fluxo de massa: pluma rasa vs. pluma profunda ao longo do dia", fontsize=15, fontweight="bold")

axs[0,0].plot(horas, [SHF(t) for t in times], color="firebrick", label="SHF")
axs[0,0].plot(horas, [LHF(t) for t in times], color="steelblue", label="LHF")
axs[0,0].set_title("Fluxos de superficie"); axs[0,0].set_ylabel("W/m^2"); axs[0,0].legend(fontsize=8)

axs[0,1].plot(horas, h, color="darkorange", label="base da nuvem (topo da CLC)")
axs[0,1].plot(horas, z_top_shallow, color="grey", label="topo da pluma rasa")
axs[0,1].plot(horas, z_top_deep, color="black", ls="--", label="alcance da pluma profunda (teste)")
axs[0,1].axhline(Z_CHECK_DEEP, color="crimson", ls=":", lw=1, label="altura de verificacao (disparo)")
axs[0,1].set_title("Alturas: CLC, topo raso, alcance da pluma profunda")
axs[0,1].set_ylabel("m"); axs[0,1].legend(fontsize=7)

axs[1,0].plot(horas, q_ml*1000, color="seagreen", label="q CLC")
axs[1,0].plot(horas, q_ft*1000, color="teal", label="q troposfera livre (camada de transicao)")
axs[1,0].set_title("Umedecimento por detranhamento REAL da pluma rasa")
axs[1,0].set_ylabel("g/kg"); axs[1,0].legend(fontsize=8)

axs[1,1].plot(horas, Mb_shallow*1000, color="purple")
axs[1,1].set_title("Fluxo de massa de base de nuvem rasa (Mb)")
axs[1,1].set_ylabel("g/m^2/s")

axs[2,0].plot(horas, survived_deep.astype(int), color="black", drawstyle="steps-post")
axs[2,0].set_title("Pluma profunda sobrevive a camada de CIN? (1=sim)")
axs[2,0].set_ylabel(""); axs[2,0].set_yticks([0,1])

axs[2,1].axis("off")
info = (
    f"eps rasa = {EPS_SHALLOW:.1e} m-1\n"
    f"eps profunda = {EPS_DEEP:.1e} m-1\n"
    f"Altura de verificacao = {Z_CHECK_DEEP:.0f} m\n\n"
    + (f"Disparo as {trigger_time/3600:.2f} h local" if trigger_time else "Sem disparo no periodo simulado")
)
axs[2,1].text(0.05, 0.5, info, fontsize=11, va="center")

axs[3,0].plot(horas, precip_shallow_rate, color="steelblue", label="chuva - regime raso")
axs[3,0].plot(horas, precip_deep_rate, color="darkred", label="chuva - regime profundo")
axs[3,0].set_title("Taxa de precipitacao de superficie (autoconversao, formulacao Tiedtke)")
axs[3,0].set_ylabel("mm/h"); axs[3,0].set_xlabel("Hora local"); axs[3,0].legend(fontsize=8)

axs[3,1].axis("off")

for ax in axs.flat:
    ax.grid(alpha=0.3)
    if trigger_time is not None:
        ax.axvline(trigger_time/3600.0, color="black", ls=":", lw=1.2)

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("fluxo_de_massa_transicao.png", dpi=160)
print("Figura salva: fluxo_de_massa_transicao.png")

# --- Figura extra: perfis verticais de empuxo e w para a pluma rasa e profunda,
#     comparando um horario de manha (sem disparo) com um horario pos-disparo ---
def plume_at_time(t_query):
    idx = int(t_query/dt)
    z_lcl_q = lcl_height(T_ml[idx], q_ml[idx])
    z0 = z_lcl_q
    T0 = T_ml[idx] - (g/cp)*z0
    q0 = qsat(T0, p_of_z(z0))
    shf = SHF(t_query)
    w_star2 = max(0.2, (g/T_ml[idx] * max(shf,1.0)/(rho0*cp) * h[idx])**(2.0/3.0) * 0.3)
    sh = run_plume(z0, T0, q0, w_star2, EPS_SHALLOW, q_ft[idx])
    dp = run_plume(z0, T0, q0, w_star2, EPS_DEEP, q_ft[idx])
    return sh, dp

t_morning = 9*3600.0
t_after   = (trigger_time + 1800.0) if trigger_time else 15*3600.0

sh_m, dp_m = plume_at_time(t_morning)
sh_a, dp_a = plume_at_time(t_after)

fig2, axs2 = plt.subplots(1, 2, figsize=(11,6), sharey=True)
axs2[0].plot(sh_m["B"], sh_m["z"], color="grey", label="pluma rasa")
axs2[0].plot(dp_m["B"], dp_m["z"], color="black", ls="--", label="pluma profunda (teste)")
axs2[0].axvline(0, color="k", lw=0.5)
axs2[0].set_title(f"Manha ({t_morning/3600:.0f}h) - antes da transicao")
axs2[0].set_xlabel("Empuxo B [m/s^2]"); axs2[0].set_ylabel("Altura [m]"); axs2[0].legend(fontsize=8)

axs2[1].plot(sh_a["B"], sh_a["z"], color="grey", label="pluma rasa")
axs2[1].plot(dp_a["B"], dp_a["z"], color="black", ls="--", label="pluma profunda (teste)")
axs2[1].axvline(0, color="k", lw=0.5)
label_after = f"{t_after/3600:.1f}h"
axs2[1].set_title(f"Apos a transicao (~{label_after})")
axs2[1].set_xlabel("Empuxo B [m/s^2]"); axs2[1].legend(fontsize=8)

for ax in axs2:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("perfis_empuxo_pluma.png", dpi=160)
print("Figura salva: perfis_empuxo_pluma.png")
