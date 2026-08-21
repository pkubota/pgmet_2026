import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""
MODELO 2D DE NUVEM CONVECTIVA -- VERSAO UNIFICADA E PARAMETRIZAVEL
=====================================================================
Curso: Conveccao Atmosferica - Regimes de Transicao

Este script junta, num unico modelo 2D explicito (vorticidade-funcao de
corrente, aproximacao de Boussinesq), a fisica que foi desenvolvida nos
modelos de coluna (transicao_rasa_profunda.py = v1,
transicao_fluxo_de_massa_v2.py = v2, transicao_fluxo_de_massa_v3.py = v3).
A diferenca central e conceitual: no modelo de coluna, processos como
entranhamento e downdraft precisam ser PARAMETRIZADOS (uma taxa epsilon,
uma funcao run_downdraft()) porque a coluna nao tem espaco para resolve-los
fisicamente. Aqui, com x e z explicitos, varios desses processos EMERGEM
da propria dinamica:

  - "Entranhamento" (v2/v3) -> aqui e a mistura turbulenta real nas bordas
    da termica ascendente (via difusao numerica/advencao).
  - "Downdraft" (v3, run_downdraft()) -> aqui NAO e mais uma funcao
    separada: a chuva cai de verdade (velocidade terminal), evapora no ar
    subsaturado abaixo da nuvem, resfria esse ar, e esse resfriamento entra
    na MESMA equacao de vorticidade que gera a ascensao -- produzindo uma
    corrente descendente fria (cold pool) que emerge sozinha, sem receita
    separada.
  - "Evaporacao sub-nuvem" (v3) -> mesma coisa: e so a evaporacao da chuva
    caindo through ar nao saturado, resolvida no proprio campo.

O QUE AINDA E PARAMETRIZADO (nao ha jeito de fugir disso num modelo deste
tamanho/resolucao):
  - Ajuste de saturacao instantaneo (condensacao/evaporacao "rapida
    demais" para a grade resolver explicitamente);
  - Autoconversao agua de nuvem -> chuva/neve (formulacao Tiedtke, igual
    ao v2/v3 -- module_cu_tiedtke_F.txt, rotina cuasc);
  - Velocidade terminal de queda de chuva/neve (constantes fixas, nao um
    espectro de gotas);
  - Difusao numerica como proxy de mistura turbulenta de subgrade.

OPCOES DE FISICA (via linha de comando -- ver secao PARAMETROS ao final
do cabecalho, ou rode com --help):
  --microfisica {nenhuma, quente, mista}
        nenhuma: dinamica seca, sem condensacao (so a termica subindo)
        quente:  agua de nuvem + chuva (fase liquida), como o v2
        mista:   + fase gelo/neve com particao por temperatura, como o v3
  --evap-chuva {on, off}
        liga/desliga a evaporacao da chuva abaixo da nuvem -- e o que
        determina se um downdraft/cold pool aparece ou nao (compare os
        dois para ver o efeito)
  --radiacao {on, off}
        liga/desliga o resfriamento radiativo lento do ambiente (v3)
  --bolha AMPLITUDE_K
        amplitude da bolha termica inicial (K) -- controla se a nuvem
        fica rasa ou rompe para profunda, como nos cenarios do v2/v3
  --tempo MINUTOS
        tempo total de simulacao (padrao 60 min; 720 min se --ciclo-diurno on)
  --ciclo-diurno {on, off}
        on: ativa a fisica de encroachment da v1/v2/v3 -- SHF(t)/LHF(t)
        senoidais, CLC crescendo (h(t), theta_ml(t), q_ml(t)), MISTURADAS
        ao perfil estatico de CIN/CAPE (para z >= h(t)). Em vez de UMA
        bolha fixa em t=0, termicas sao disparadas periodicamente com
        amplitude escalada pelo SHF(t) do momento -- o disparo da
        conveccao profunda passa a ser uma CONSEQUENCIA do ciclo diurno,
        nao um cenario prescrito manualmente. Sem isso (off, padrao),
        o modelo roda como um "estudo de caso" de uma unica termica num
        ambiente fixo -- mais rapido, bom para comparar --bolha fraca vs.
        forte isoladamente.
  --cenario NOME
        rotulo usado nos nomes dos arquivos de figura gerados

EXEMPLOS:
  python3 nuvem_2d.py --bolha 2.5 --microfisica quente --cenario rasa
  python3 nuvem_2d.py --bolha 7.0 --microfisica mista --evap-chuva on --cenario profunda_downdraft
  python3 nuvem_2d.py --bolha 7.0 --microfisica mista --evap-chuva off --cenario profunda_sem_downdraft
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# PARAMETROS (linha de comando)
# =====================================================================
parser = argparse.ArgumentParser(description="Modelo 2D de nuvem convectiva -- versao parametrizavel")
parser.add_argument("--bolha", type=float, default=3.0, help="amplitude da bolha termica inicial [K]")
parser.add_argument("--cenario", type=str, default="bolha", help="rotulo p/ os arquivos de saida")
parser.add_argument("--microfisica", choices=["nenhuma", "quente", "mista"], default="mista",
                     help="nenhuma=dinamica seca; quente=agua+chuva (v2); mista=+gelo/neve (v3)")
parser.add_argument("--evap-chuva", choices=["on", "off"], default="on",
                     help="evaporacao da chuva abaixo da nuvem (gera downdraft/cold pool se 'on')")
parser.add_argument("--radiacao", choices=["on", "off"], default="off",
                     help="resfriamento radiativo lento do ambiente (v3)")
parser.add_argument("--ciclo-diurno", choices=["on", "off"], default="off",
                     help="ativa SHF/LHF diurnos + CLC crescendo (encroachment, v1/v2/v3)")
parser.add_argument("--tempo", type=float, default=None, help="tempo total de simulacao [min]")
args, _unknown = parser.parse_known_args()

MICROFISICA = args.microfisica
EVAP_CHUVA = (args.evap_chuva == "on")
RADIACAO = (args.radiacao == "on")
CICLO_DIURNO = (args.ciclo_diurno == "on")
DTHETA_BUBBLE = args.bolha
CENARIO = args.cenario
if args.tempo is not None:
    T_END = args.tempo * 60.0
else:
    T_END = 720.0*60.0 if CICLO_DIURNO else 60.0*60.0

print(f"Configuracao: bolha={DTHETA_BUBBLE}K  microfisica={MICROFISICA}  "
      f"evap_chuva={EVAP_CHUVA}  radiacao={RADIACAO}  ciclo_diurno={CICLO_DIURNO}  "
      f"tempo={T_END/60:.0f}min  cenario={CENARIO}")

# =====================================================================
# 1. CONSTANTES E GRADE
# =====================================================================
g = 9.81
cp = 1004.0
Rd = 287.0
Rv = 461.5
Lv = 2.5e6
Lf = 3.337e5
Ls = Lv + Lf
eps_R = Rd/Rv
p0 = 1000.0
theta0 = 300.0

nx, nz = 90, 110
dx, dz = 100.0, 100.0
x = np.arange(nx)*dx
z = np.arange(nz)*dz
X, Z = np.meshgrid(x, z, indexing='ij')

def p_of_z(zz, H=8000.0):
    return p0*np.exp(-zz/H)

def exner(zz):
    return (p_of_z(zz)/p0)**(Rd/cp)

def qsat_liq(T, p):
    Tc = T - 273.15
    es = 6.112*np.exp(17.67*Tc/(Tc+243.5))
    return eps_R*es/np.maximum(p-es, 1e-3)

def qsat_ice(T, p):
    Tc = T - 273.15
    es = 6.112*np.exp(22.46*Tc/(Tc+272.62))
    return eps_R*es/np.maximum(p-es, 1e-3)

T_ICE_ALL = 233.15
T_LIQ_ALL = 273.15

def ice_fraction(T):
    f = (T_LIQ_ALL - T)/(T_LIQ_ALL - T_ICE_ALL)
    return np.clip(f, 0.0, 1.0)

def qsat_mixed(T, p):
    fi = ice_fraction(T)
    return (1.0-fi)*qsat_liq(T, p) + fi*qsat_ice(T, p)

# =====================================================================
# 1b. FORCANTE DIURNA E CLC (encroachment) -- so usado se --ciclo-diurno on
#     Mesmas formulas de transicao_rasa_profunda.py (v1): SHF/LHF senoidais,
#     crescimento da CLC por energia (dh/dt = SHF/(rho*cp*h*gamma_local)).
# =====================================================================
t_sunrise = 6*3600.0    # soh para exibicao (hora = 6 + t/3600); a simulacao ja COMECA no nascer do sol
daylength = 12*3600.0
SHF_max, LHF_max = 250.0, 300.0
rho0_sfc = 1.15

def SHF(t):
    """t = segundos DESDE O INICIO DA SIMULACAO (que ja comeca as 6h)."""
    return max(0.0, SHF_max*np.sin(np.pi*t/daylength)) if 0 <= t <= daylength else 0.0

def LHF(t):
    """t = segundos DESDE O INICIO DA SIMULACAO (que ja comeca as 6h)."""
    return max(0.0, LHF_max*np.sin(np.pi*t/daylength)) if 0 <= t <= daylength else 0.0

# Estado da CLC (atualizado a cada passo se CICLO_DIURNO=True)
h_clc = 200.0        # profundidade inicial da CLC [m]
theta_ml = None      # diagnosticada (encroachment: theta_ml = theta_env_static(h_clc))
q_ml = 11.0e-3        # umidade especifica inicial da CLC [kg/kg]

REARM_INTERVAL = 600.0    # s -- intervalo entre disparos de novas termicas (10 min)
BASE_BUBBLE_AMP = 4.0      # K -- amplitude de referencia (escalada por SHF(t)/SHF_max)
_last_trigger_t = [-1e9]   # mutavel: hora do ultimo disparo de termica

# =====================================================================
# 2. ESTADO BASE (ambiente) -- mesma filosofia dos modelos de coluna:
#    camada quase-neutra -> capa estavel (CIN) -> camada fracamente
#    estavel em theta mas condicionalmente instavel na saturacao (CAPE)
#    -> capa estavel no topo
# =====================================================================
def dtheta_dz_env(zz):
    return np.where(zz < 1000, 3.0e-3,
           np.where(zz < 2000, 6.5e-3,
           np.where(zz < 8500, 2.0e-3,
                    6.0e-3)))

theta_env_1d = np.zeros(nz)
theta_env_1d[0] = theta0
for k in range(1, nz):
    theta_env_1d[k] = theta_env_1d[k-1] + dtheta_dz_env(z[k-1])*dz

PI = exner(z)
P = p_of_z(z)
T_env_1d = theta_env_1d*PI
QSAT_ENV_1d = qsat_liq(T_env_1d, P)

def RH_env_profile(zz):
    return np.where(zz < 1000, 0.70,
           np.where(zz < 2000, 0.35,
           np.where(zz < 8500, 0.55,
                    0.20)))

RH_env_1d = RH_env_profile(z)
qv_env_1d = RH_env_1d*QSAT_ENV_1d

THETA_ENV = np.broadcast_to(theta_env_1d[None, :], (nx, nz)).copy()
DTHETA_ENV_DZ = np.gradient(theta_env_1d, dz)
DQV_ENV_DZ = np.gradient(qv_env_1d, dz)
QV_ENV_BASE = qv_env_1d.copy()   # referencia p/ resfriamento radiativo nao mexer na umidade

RAD_COOL_RATE = 1.5/86400.0   # K/s (~1.5 K/dia) -- so usado se --radiacao on

def build_env_now(h_now, theta_ml_now, q_ml_now):
    """
    Perfil ambiente instantaneo: para z<h_now, a CLC bem misturada
    (theta_ml_now, q_ml_now, constantes com a altura); para z>=h_now, o
    perfil estatico de fundo (theta_env_1d/qv_env_1d, a mesma estrutura
    de CIN/CAPE de sempre). Por construcao do encroachment, theta_ml_now
    == theta_env_1d no nivel h_now, entao o perfil de theta emenda sem
    salto ali; q_ml_now pode ser diferente de qv_env_1d(h_now) (o mesmo
    "entranhamento seca a CLC" da v1/v2/v3), entao a umidade PODE ter um
    pequeno salto em z=h_now -- fisicamente esperado (a base da nuvem
    real quase sempre tem um salto de umidade no topo da CLC).
    """
    theta_now = np.where(z < h_now, theta_ml_now, theta_env_1d)
    qv_now = np.where(z < h_now, q_ml_now, qv_env_1d)
    dtheta_dz_now = np.gradient(theta_now, dz)
    dqv_dz_now = np.gradient(qv_now, dz)
    return theta_now, qv_now, dtheta_dz_now, dqv_dz_now

# =====================================================================
# 3. CAMPOS PROGNOSTICOS (criados conforme a opcao de microfisica)
# =====================================================================
zeta = np.zeros((nx, nz))
psi = np.zeros((nx, nz))
thp = np.zeros((nx, nz))
qvp = np.zeros((nx, nz))
qc = np.zeros((nx, nz))
qi = np.zeros((nx, nz))              # so usado se microfisica == "mista"
qr = np.zeros((nx, nz))              # chuva -- usado se microfisica != "nenhuma"
qsnow = np.zeros((nx, nz))           # neve/graupel -- so usado se microfisica == "mista"

surface_rain_accum = np.zeros(nx)     # acumulado de chuva na superficie [mm]

# --- Bolha termica inicial (o gatilho) -- so no modo sem ciclo diurno ---
X0, Z0 = 4500.0, 500.0
RX, RZ = 1100.0, 600.0
r2 = ((X-X0)/RX)**2 + ((Z-Z0)/RZ)**2
if not CICLO_DIURNO:
    thp += DTHETA_BUBBLE*np.exp(-r2)*(r2 < 6)
    qvp += 0.5e-3*np.exp(-r2)*(r2 < 6)

def dispara_termica(t_now, h_now, amp_scale):
    """
    Modo --ciclo-diurno: em vez de uma unica bolha em t=0, novas termicas
    sao injetadas periodicamente (a cada REARM_INTERVAL) partindo do topo
    da CLC (h_now), com amplitude escalada pelo aquecimento de superficie
    do momento (amp_scale = SHF(t)/SHF_max). Representa, de forma muito
    simplificada, a populacao continua de termicas de camada limite --
    nao um espectro real, so um gatilho periodico fisicamente motivado.
    """
    global thp, qvp
    if t_now - _last_trigger_t[0] < REARM_INTERVAL:
        return
    _last_trigger_t[0] = t_now
    amp = BASE_BUBBLE_AMP*max(amp_scale, 0.05)
    z0_bubble = max(h_now*0.5, 100.0)
    rz_bubble = max(h_now*0.4, 150.0)
    r2_now = ((X-X0)/RX)**2 + ((Z-z0_bubble)/rz_bubble)**2
    thp[:] = thp + amp*np.exp(-r2_now)*(r2_now < 6)
    qvp[:] = qvp + 0.5e-3*np.exp(-r2_now)*(r2_now < 6)

# =====================================================================
# 4. OPERADORES NUMERICOS
# =====================================================================
def poisson_jacobi(zeta_field, psi_guess, niter=120, omega=1.0):
    psi_f = psi_guess.copy()
    dx2, dz2 = dx*dx, dz*dz
    denom = 2.0*(1.0/dx2 + 1.0/dz2)
    for _ in range(niter):
        rhs = ((psi_f[2:, 1:-1]+psi_f[:-2, 1:-1])/dx2 +
               (psi_f[1:-1, 2:]+psi_f[1:-1, :-2])/dz2 - zeta_field[1:-1, 1:-1])
        psi_interior_new = rhs/denom
        psi_f[1:-1, 1:-1] = (1-omega)*psi_f[1:-1, 1:-1] + omega*psi_interior_new
    psi_f[0, :] = 0.0; psi_f[-1, :] = 0.0
    psi_f[:, 0] = 0.0; psi_f[:, -1] = 0.0
    return psi_f

def velocities(psi_f):
    u = np.zeros_like(psi_f)
    w = np.zeros_like(psi_f)
    u[:, 1:-1] = -(psi_f[:, 2:]-psi_f[:, :-2])/(2*dz)
    w[1:-1, :] = (psi_f[2:, :]-psi_f[:-2, :])/(2*dx)
    return u, w

def upwind_advect(f, u, w):
    """Adveccao upwind de 1a ordem. 'w' pode ja incluir uma velocidade de
    queda (para chuva/neve): passe (w_ar - Vt) em vez de w_ar."""
    dfdx = np.zeros_like(f)
    dfdz = np.zeros_like(f)
    dfdx[1:-1, :] = np.where(u[1:-1, :] > 0,
                              (f[1:-1, :]-f[:-2, :])/dx,
                              (f[2:, :]-f[1:-1, :])/dx)
    dfdz[:, 1:-1] = np.where(w[:, 1:-1] > 0,
                              (f[:, 1:-1]-f[:, :-2])/dz,
                              (f[:, 2:]-f[:, 1:-1])/dz)
    return -(u*dfdx + w*dfdz)

def laplacian(f):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = ((f[2:, 1:-1]-2*f[1:-1, 1:-1]+f[:-2, 1:-1])/dx**2 +
                        (f[1:-1, 2:]-2*f[1:-1, 1:-1]+f[1:-1, :-2])/dz**2)
    return lap

def apply_bc(f, zero_grad=True):
    if zero_grad:
        f[0, :] = f[1, :]; f[-1, :] = f[-2, :]
        f[:, 0] = f[:, 1]; f[:, -1] = f[:, -2]
    else:
        f[0, :] = 0.0; f[-1, :] = 0.0; f[:, 0] = 0.0; f[:, -1] = 0.0
    return f

K_DIFF = 25.0

# --- Autoconversao (relaxacao tipo Kessler acima de um limiar -- mesma forma
#     matematica usada em run_plume() do v2/v3; a versao com profundidade
#     acima da base da nuvem do Tiedtke/v2/v3 exigiria rastrear a altura da
#     base por coluna, o que fica como extensao possivel) ---
QC_CRIT = 0.3e-3              # kg/kg -- limiar de autoconversao
TAU_AUTO = 300.0              # s
TAU_AUTO_ICE = 550.0           # s -- mais lento (agregacao/deposicao vs. coalescencia)

# --- Velocidades terminais de queda (constantes simplificadas) ---
VT_RAIN = 5.0    # m/s
VT_SNOW = 1.5    # m/s

# --- Evaporacao da chuva/neve no ar nao saturado ---
EVAP_COEF = 2.0e-3   # 1/s por unidade de deficit de saturacao (kg/kg)

# =====================================================================
# 5. LOOP TEMPORAL
# =====================================================================
dt = 1.5
nsteps = int(T_END/dt)
save_every = int(300/dt)

frames_qc, frames_qr, frames_w, frames_thp, frames_t = [], [], [], [], []
frames_qvp, frames_u = [], []   # necessarios para calcular as tendencias (dtheta/dt, dq/dt, du/dt)
frames_h_clc, frames_SHF = [], []   # diagnostico do ciclo diurno (so relevante se CICLO_DIURNO)

for step in range(nsteps+1):
    u, w = velocities(psi)
    t_now = step*dt

    # --- ciclo diurno (opcional): CLC cresce por encroachment, dispara termicas ---
    if CICLO_DIURNO:
        shf_now = SHF(t_now)
        lhf_now = LHF(t_now)
        if theta_ml is None:
            theta_ml = float(np.interp(h_clc, z, theta_env_1d))
        gamma_local = max(float(np.interp(h_clc, z, np.gradient(theta_env_1d, dz))), 1.0e-4)
        dh = dt*shf_now/(rho0_sfc*cp*h_clc*gamma_local) if shf_now > 0 else 0.0
        h_clc = max(h_clc + dh, 200.0)
        theta_ml = float(np.interp(h_clc, z, theta_env_1d))
        q_env_top = float(np.interp(h_clc, z, qv_env_1d))
        q_ml = q_ml + dt*(lhf_now/(rho0_sfc*Lv*h_clc)) + (q_env_top - q_ml)*(dh/h_clc if h_clc > 0 else 0.0)
        q_ml = max(q_ml, 1.0e-4)

        theta_env_now_1d, qv_env_now_1d, dtheta_dz_now_1d, dqv_dz_now_1d = build_env_now(h_clc, theta_ml, q_ml)

        if shf_now > 0:
            dispara_termica(t_now, h_clc, shf_now/SHF_max)

        if step % save_every == 0:
            frames_h_clc.append(h_clc)
            frames_SHF.append(shf_now)
    else:
        theta_env_now_1d, qv_env_now_1d = theta_env_1d, qv_env_1d
        dtheta_dz_now_1d, dqv_dz_now_1d = DTHETA_ENV_DZ, DQV_ENV_DZ

    # --- resfriamento radiativo lento (opcional) ---
    if RADIACAO:
        THETA_ENV_now = theta_env_now_1d[None, :] - RAD_COOL_RATE*t_now
    else:
        THETA_ENV_now = theta_env_now_1d[None, :]

    T = (THETA_ENV_now+thp)*PI[None, :]

    # --- ajuste de saturacao (condensacao/evaporacao instantanea) ---
    if MICROFISICA != "nenhuma":
        if MICROFISICA == "mista":
            qs = qsat_mixed(T, P[None, :])
            fi = ice_fraction(T)
        else:
            qs = qsat_liq(T, P[None, :])
            fi = np.zeros_like(T)

        qv_total = qv_env_now_1d[None, :] + qvp
        cond = qv_total - qs
        to_condense = np.maximum(cond, 0.0)
        # evapora primeiro a agua liquida de nuvem, depois (mista) o gelo
        to_evap_liq = np.minimum(np.maximum(-cond, 0.0), qc)
        resto = np.maximum(-cond, 0.0) - to_evap_liq
        to_evap_ice = np.minimum(resto, qi) if MICROFISICA == "mista" else 0.0

        d_ql = to_condense*(1.0-fi) - to_evap_liq
        d_qi = (to_condense*fi - to_evap_ice) if MICROFISICA == "mista" else 0.0

        qvp += -(to_condense) + to_evap_liq + (to_evap_ice if MICROFISICA == "mista" else 0.0)
        qc += d_ql
        if MICROFISICA == "mista":
            qi += d_qi
        Leff_cond = Lv*(to_condense*(1.0-fi)) + (Ls*(to_condense*fi) if MICROFISICA == "mista" else 0.0)
        Leff_evap = Lv*to_evap_liq + (Ls*to_evap_ice if MICROFISICA == "mista" else 0.0)
        thp += (Leff_cond - Leff_evap)/(cp*PI[None, :])

        # --- autoconversao: agua/gelo de nuvem -> chuva/neve (Tiedtke) ---
        excess_l = np.maximum(qc-QC_CRIT, 0.0)
        removed_l = excess_l*(1-np.exp(-dt/TAU_AUTO))
        qc -= removed_l
        qr += removed_l

        if MICROFISICA == "mista":
            excess_i = np.maximum(qi-QC_CRIT, 0.0)
            removed_i = excess_i*(1-np.exp(-dt/TAU_AUTO_ICE))
            qi -= removed_i
            qsnow += removed_i

            # --- derretimento de neve abaixo do nivel de congelamento ---
            melt_mask = (T > T_LIQ_ALL) & (qsnow > 0)
            melted = np.where(melt_mask, qsnow, 0.0)
            qsnow -= melted
            qr += melted
            thp -= (Lf*melted)/(cp*PI[None, :])   # derreter consome calor latente de fusao

        # --- evaporacao da chuva/neve no ar nao saturado (=> downdraft resolvido) ---
        if EVAP_CHUVA:
            qs_now = qsat_liq(T, P[None, :])
            deficit = np.maximum(qs_now - (qv_env_now_1d[None, :]+qvp), 0.0)
            evap_rain = np.minimum(qr, EVAP_COEF*dt*deficit*1000.0*qr)
            qr -= evap_rain
            qvp += evap_rain
            thp -= (Lv*evap_rain)/(cp*PI[None, :])   # resfriamento evaporativo -> puxa o downdraft

            if MICROFISICA == "mista":
                evap_snow = np.minimum(qsnow, EVAP_COEF*dt*deficit*1000.0*qsnow)
                qsnow -= evap_snow
                qvp += evap_snow
                thp -= (Ls*evap_snow)/(cp*PI[None, :])

    # (nota: um diagnostico fisico de taxa de precipitacao em mm/h exigiria
    #  integrar o fluxo de massa vertical qr*Vt_rain na ultima camada antes
    #  do fundo do dominio -- deixado como exercicio de extensao; aqui o
    #  foco e no campo qr em si e no efeito dinamico do downdraft, nao no
    #  acumulado de superficie)

    if step == nsteps:
        break

    # --- tendencias dinamicas ---
    thv = thp + 0.61*theta0*qvp - theta0*(qc+qi+qr+qsnow)   # empuxo com carregamento de agua/gelo
    dthv_dx = np.zeros_like(thp)
    dthv_dx[1:-1, :] = (thv[2:, :]-thv[:-2, :])/(2*dx)
    buoy_torque = (g/theta0)*dthv_dx

    dzeta = upwind_advect(zeta, u, w) + buoy_torque + K_DIFF*laplacian(zeta)
    dthp = upwind_advect(thp, u, w) - w*dtheta_dz_now_1d[None, :] + K_DIFF*laplacian(thp)
    dqvp = upwind_advect(qvp, u, w) - w*dqv_dz_now_1d[None, :] + K_DIFF*laplacian(qvp)
    dqc = upwind_advect(qc, u, w) + K_DIFF*laplacian(qc)
    dqi = upwind_advect(qi, u, w) + K_DIFF*laplacian(qi) if MICROFISICA == "mista" else 0.0
    dqr = upwind_advect(qr, u, w-VT_RAIN) + K_DIFF*laplacian(qr) if MICROFISICA != "nenhuma" else 0.0
    dqsnow = upwind_advect(qsnow, u, w-VT_SNOW) + K_DIFF*laplacian(qsnow) if MICROFISICA == "mista" else 0.0

    zeta = zeta + dt*dzeta
    thp = thp + dt*dthp
    qvp = qvp + dt*dqvp
    qc = np.maximum(qc + dt*dqc, 0.0)
    if MICROFISICA == "mista":
        qi = np.maximum(qi + dt*dqi, 0.0)
    if MICROFISICA != "nenhuma":
        qr = np.maximum(qr + dt*dqr, 0.0)
    if MICROFISICA == "mista":
        qsnow = np.maximum(qsnow + dt*dqsnow, 0.0)

    zeta = apply_bc(zeta, zero_grad=False)
    thp = apply_bc(thp, zero_grad=True)
    qvp = apply_bc(qvp, zero_grad=True)
    qc = apply_bc(qc, zero_grad=True)
    if MICROFISICA == "mista":
        qi = apply_bc(qi, zero_grad=True)
    if MICROFISICA != "nenhuma":
        qr = apply_bc(qr, zero_grad=True)
    if MICROFISICA == "mista":
        qsnow = apply_bc(qsnow, zero_grad=True)

    psi = poisson_jacobi(zeta, psi, niter=120)

    if step % save_every == 0:
        frames_qc.append((qc+qi).copy())
        frames_qr.append((qr+qsnow).copy())
        frames_w.append(w.copy())
        frames_thp.append(thp.copy())
        frames_qvp.append(qvp.copy())
        frames_u.append(u.copy())
        frames_t.append(step*dt)
        cloud_top = z[np.where((qc+qi).max(axis=0) > 1e-5)[0].max()] if (qc+qi).max() > 1e-5 else 0
        extra = f" | hora={6+t_now/3600:5.2f}h | h_CLC={h_clc:5.0f}m | SHF={SHF(t_now):5.0f}W/m2" if CICLO_DIURNO else ""
        print(f"t={step*dt/60:5.1f} min | w_max={w.max():5.2f} m/s | w_min={w.min():5.2f} m/s | "
              f"qc_max={(qc+qi).max()*1000:5.3f} g/kg | topo_nuvem~{cloud_top:6.0f} m | "
              f"thp_min={thp.min():5.2f} K (cold pool se negativo perto do chao){extra}")

print("Simulacao concluida.")

# =====================================================================
# 6. FIGURAS
# =====================================================================
# --- Tendencias: diferenca entre quadros consecutivos (dt = save_every*dt) ---
# No modelo de coluna (v3), a tendencia tem que ser PARAMETRIZADA (funcao
# calcula_tendencias(), subsidencia compensatoria + pulso de detranhamento)
# porque a coluna nao resolve x. Aqui NAO -- e so a diferenca no tempo do
# proprio campo resolvido: dtheta/dt = (theta'(t+dt)-theta'(t))/dt. Mesma
# fisica (slide 39 do curso), mas sem precisar de nenhuma parametrizacao.
frame_dt_days = (frames_t[1]-frames_t[0])/86400.0 if len(frames_t) > 1 else 1.0
frames_dtheta_dt = []   # K/dia
frames_dq_dt = []       # g/kg/dia
frames_du_dt = []       # m/s/dia
for j in range(len(frames_t)-1):
    frames_dtheta_dt.append((frames_thp[j+1]-frames_thp[j])/frame_dt_days)
    frames_dq_dt.append((frames_qvp[j+1]-frames_qvp[j])/frame_dt_days*1000.0)
    frames_du_dt.append((frames_u[j+1]-frames_u[j])/frame_dt_days)
# o ultimo quadro nao tem "proximo" -- repete a ultima tendencia valida so
# para manter os paineis alinhados com os das linhas de estado
frames_dtheta_dt.append(frames_dtheta_dt[-1] if frames_dtheta_dt else np.zeros_like(thp))
frames_dq_dt.append(frames_dq_dt[-1] if frames_dq_dt else np.zeros_like(thp))
frames_du_dt.append(frames_du_dt[-1] if frames_du_dt else np.zeros_like(thp))

n_panels = min(6, len(frames_qc))
idxs = np.linspace(0, len(frames_qc)-1, n_panels).astype(int)

fig, axs = plt.subplots(6, n_panels, figsize=(3.1*n_panels, 19), sharex=True, sharey=True)
for j, idx in enumerate(idxs):
    t_min = frames_t[idx]/60
    ax1 = axs[0, j]
    ax1.contourf(X/1000, Z/1000, frames_qc[idx]*1000, levels=np.linspace(0, 2, 11), cmap="Blues", extend="max")
    ax1.set_title(f"t={t_min:.0f} min\nqc+qi [g/kg]", fontsize=9)
    if j == 0: ax1.set_ylabel("altura [km]")

    ax2 = axs[1, j]
    lim = max(1.0, np.abs(frames_w[idx]).max())
    ax2.contourf(X/1000, Z/1000, frames_w[idx], levels=np.linspace(-lim, lim, 15), cmap="RdBu_r")
    ax2.set_title("w [m/s]", fontsize=9)
    if j == 0: ax2.set_ylabel("altura [km]")

    ax3 = axs[2, j]
    qr_max = max(1e-4, frames_qr[idx].max())
    ax3.contourf(X/1000, Z/1000, frames_qr[idx]*1000, levels=np.linspace(0, qr_max*1000, 11),
                 cmap="Greens", extend="max")
    ax3.set_title("chuva+neve [g/kg]", fontsize=9)
    if j == 0: ax3.set_ylabel("altura [km]")

    ax4 = axs[3, j]
    lim_th = max(5.0, np.abs(frames_dtheta_dt[idx]).max())
    ax4.contourf(X/1000, Z/1000, frames_dtheta_dt[idx], levels=np.linspace(-lim_th, lim_th, 15),
                 cmap="RdBu_r")
    ax4.set_title("dtheta/dt [K/dia]", fontsize=9)
    if j == 0: ax4.set_ylabel("altura [km]")

    ax5 = axs[4, j]
    lim_q = max(5.0, np.abs(frames_dq_dt[idx]).max())
    ax5.contourf(X/1000, Z/1000, frames_dq_dt[idx], levels=np.linspace(-lim_q, lim_q, 15),
                 cmap="BrBG")
    ax5.set_title("dq/dt [g/kg/dia]", fontsize=9)
    if j == 0: ax5.set_ylabel("altura [km]")

    ax6 = axs[5, j]
    lim_u = max(5.0, np.abs(frames_du_dt[idx]).max())
    ax6.contourf(X/1000, Z/1000, frames_du_dt[idx], levels=np.linspace(-lim_u, lim_u, 15),
                 cmap="PuOr")
    ax6.set_title("du/dt [m/s/dia]", fontsize=9)
    ax6.set_xlabel("x [km]")
    if j == 0: ax6.set_ylabel("altura [km]")

fig.suptitle(f"Modelo 2D unificado -- {CENARIO} (bolha={DTHETA_BUBBLE}K, microfisica={MICROFISICA}, "
             f"evap_chuva={EVAP_CHUVA}, radiacao={RADIACAO})", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"nuvem_2d_evolucao_{CENARIO}.png", dpi=150)
print(f"Figura salva: nuvem_2d_evolucao_{CENARIO}.png")

# --- Figura extra: perfil de theta' minimo perto do chao (diagnostico de cold pool) ---
if len(frames_thp) > 0:
    thp_sfc = [f[:, 1].min() for f in frames_thp]
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.array(frames_t)/60, thp_sfc, marker="o")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("tempo [min]")
    ax.set_ylabel("theta' minimo perto do chao [K]")
    ax.set_title(f"Diagnostico de cold pool -- {CENARIO}\n(negativo = ar frio de downdraft chegando a superficie)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"nuvem_2d_coldpool_{CENARIO}.png", dpi=150)
    print(f"Figura salva: nuvem_2d_coldpool_{CENARIO}.png")
