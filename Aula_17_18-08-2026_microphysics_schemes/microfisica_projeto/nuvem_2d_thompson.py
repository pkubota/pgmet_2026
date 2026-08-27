import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""
MODELO 2D DE NUVEM CONVECTIVA -- ACOPLADO AO ESQUEMA THOMPSON COMPLETO
=========================================================================
Curso: Conveccao Atmosferica / Microfisica de Nuvens (MET-756-4)

Esta e uma variante de `nuvem_2d.py` (modelo 2D explicito, vorticidade-
funcao de corrente, aproximacao de Boussinesq -- ver Equacao 2.6 do
relatorio "Modelo Conceitual de Fluxo de Massa para a Transicao
Convectiva Rasa-Profunda") em que o bloco de microfisica simplificado
(ajuste de saturacao de 1 momento, autoconversao por limiar unico,
velocidades terminais fixas) foi SUBSTITUIDO pelo esquema completo de
6 categorias desenvolvido no projeto de microfisica (Passos 1-3):

    qv, qc(+Nc), qr(+Nr), qi(+Ni), qs(+Ns), qg(+Ng)

com todos os processos: Pccnd, Pccnr, Pracw, Pr_self, Prevp (chuva
quente), Pidsn, Pidep, Pifzc, Pimlt, Pi_iacw (gelo), Psdep, Pgdep,
Pgfzr, riming (Ps_sacw/Pgacw), Hallett-Mossop (Pispl), coleta de chuva
(Ps_sacr/Pgacr/Piacr), Picns e degelo de neve/graupel (fase mista).
Ver `microfisica/coluna_generica.py` para a funcao que encapsula essa
fisica de forma independente de qualquer classe de coluna.

O NUCLEO DINAMICO (Equacao 2.6) NAO MUDA
-------------------------------------------
    u = -d(psi)/dz,  w = d(psi)/dx,  Laplaciano(psi) = zeta
    d(zeta)/dt = -v.grad(zeta) + (g/theta0)*d(theta_v')/dx + K*Laplaciano(zeta)

A microfisica entra em EXATAMENTE dois pontos, como no modelo original:
    1. Fonte/sumidouro de qv,qc,qr,qi,qs,qg e de calor latente (thp) --
       agora calculado pela funcao `passo_microfisica_coluna()`, coluna
       a coluna (loop em x), em vez do ajuste de saturacao simplificado;
    2. O termo de empuxo (thv), que agora soma TODAS as 5 categorias de
       agua condensada (qc+qi+qr+qs+qg), nao so qc+qi+qr+qsnow.

CUSTO COMPUTACIONAL (leia antes de rodar!)
---------------------------------------------
A fisica completa (10 grupos de processos, varios com ramificacoes) e
resolvida com um loop Python explicito por NIVEL, dentro de um loop por
COLUNA (x), a cada passo de tempo -- ao contrario do resto do modelo
(dinamica), que e vetorizado com NumPy sobre a grade inteira. Isso foi
uma escolha DELIBERADA (fidelidade fisica > velocidade, ver discussao
no chat), mas tem um custo real: para a grade padrao (nx=90, nz=110) e
dt~1.5s, cada minuto simulado custa muito mais tempo de CPU que no
`nuvem_2d.py` original. RECOMENDACOES:
    - Para testes/depuracao, use `--tempo` pequeno (5-10 min) e/ou
      reduza nx/nz abaixo antes de rodar o cenario completo;
    - As velocidades terminais de sedimentacao (Vt de qr,qi,qs,qg) SAO
      vetorizadas com NumPy (formulas algebricas simples, sem
      ramificacao) -- so a parte de REACOES quimicas/microfisicas usa
      o loop Python.
OPCOES DE FISICA (via linha de comando -- ver secao PARAMETROS ao final
do cabecalho, ou rode com --help):
  --microfisica {"nenhuma", "thompson"}
        nenhuma: dinamica seca, sem condensacao (so a termica subindo)
        thompson=esquema completo de 6 categorias (Passo 3)")
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

  --nx", type=int, default=90, help="pontos de grade na horizontal"
  --nz", type=int, default=110, help="pontos de grade na vertical"

EXEMPLOS:
  python3 nuvem_2d.py --bolha 2.5 --microfisica thompson 
  python3 nuvem_2d.py --bolha 7.0 --microfisica thompson --evap-chuva on 
  python3 nuvem_2d.py --bolha 7.0 --microfisica thompson --evap-chuva off 
"""


import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microfisica.constantes import (
    Rd, cp, Lv, Ls, Lf, rho_w, rho_i, rho_s, rho_g,
    MU_RAIN, MU_ICE, MU_SNOW, QMIN, NMIN, gamma_func,
)
from microfisica.coluna_generica import passo_microfisica_coluna

# =====================================================================
# PARAMETROS (linha de comando)
# =====================================================================
parser = argparse.ArgumentParser(description="Modelo 2D de nuvem convectiva -- acoplado ao esquema Thompson completo")
parser.add_argument("--bolha", type=float, default=3.0, help="amplitude da bolha termica inicial [K]")
parser.add_argument("--cenario", type=str, default="bolha", help="rotulo p/ os arquivos de saida")
parser.add_argument("--microfisica", choices=["nenhuma", "thompson"], default="thompson",
                     help="nenhuma=dinamica seca; thompson=esquema completo de 6 categorias (Passo 3)")
parser.add_argument("--evap-chuva", choices=["on", "off"], default="on",
                     help="evaporacao da chuva abaixo da nuvem (gera downdraft/cold pool se 'on')")
parser.add_argument("--radiacao", choices=["on", "off"], default="off",
                     help="resfriamento radiativo lento do ambiente")
parser.add_argument("--ciclo-diurno", choices=["on", "off"], default="off",
                     help="ativa SHF/LHF diurnos + CLC crescendo (encroachment)")
parser.add_argument("--tempo", type=float, default=None, help="tempo total de simulacao [min]")
parser.add_argument("--nx", type=int, default=90, help="pontos de grade na horizontal")
parser.add_argument("--nz", type=int, default=110, help="pontos de grade na vertical")
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
      f"tempo={T_END/60:.0f}min  cenario={CENARIO}  nx={args.nx}  nz={args.nz}")
if MICROFISICA == "thompson":
    print("AVISO: microfisica='thompson' usa o esquema completo (Passo 3, 10 grupos de "
          "processos) resolvido coluna a coluna -- bem mais lento que o esquema simplificado "
          "do nuvem_2d.py original. Para testes rapidos, use --tempo pequeno (ex.: --tempo 10).")

# =====================================================================
# 1. CONSTANTES E GRADE
# =====================================================================
g = 9.81
Rv = 461.5
eps_R = Rd/Rv
p0 = 1000.0
theta0 = 300.0

nx, nz = args.nx, args.nz
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

# =====================================================================
# 1b. FORCANTE DIURNA E CLC (encroachment) -- so usado se --ciclo-diurno on
# =====================================================================
t_sunrise = 6*3600.0
daylength = 12*3600.0
SHF_max, LHF_max = 250.0, 300.0
rho0_sfc = 1.15

def SHF(t):
    return max(0.0, SHF_max*np.sin(np.pi*t/daylength)) if 0 <= t <= daylength else 0.0

def LHF(t):
    return max(0.0, LHF_max*np.sin(np.pi*t/daylength)) if 0 <= t <= daylength else 0.0

h_clc = 200.0
theta_ml = None
q_ml = 11.0e-3

REARM_INTERVAL = 600.0
BASE_BUBBLE_AMP = 4.0
_last_trigger_t = [-1e9]

# =====================================================================
# 2. ESTADO BASE (ambiente)
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

# densidade do ar de REFERENCIA (perfil ambiente, fixo -- consistente com
# a aproximacao de Boussinesq: variacoes de densidade so entram no termo
# de empuxo, nao nas conversoes de unidade da microfisica)
RHO0_1d = P*100.0/(Rd*T_env_1d)   # P em hPa -> Pa (fator 100)
RHO0_2d = np.broadcast_to(RHO0_1d[None, :], (nx, nz)).copy()
P_PA_1d = P*100.0  # pressao em Pa, para as funcoes de microfisica
P_PA_2d = np.broadcast_to(P_PA_1d[None, :], (nx, nz)).copy()

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
QV_ENV_BASE = qv_env_1d.copy()

RAD_COOL_RATE = 1.5/86400.0

def build_env_now(h_now, theta_ml_now, q_ml_now):
    theta_now = np.where(z < h_now, theta_ml_now, theta_env_1d)
    qv_now = np.where(z < h_now, q_ml_now, qv_env_1d)
    dtheta_dz_now = np.gradient(theta_now, dz)
    dqv_dz_now = np.gradient(qv_now, dz)
    return theta_now, qv_now, dtheta_dz_now, dqv_dz_now

# =====================================================================
# 3. CAMPOS PROGNOSTICOS
#    (qc,Nc,qr,Nr,qi,Ni,qs,Ns,qg,Ng -- as 5 categorias de agua do
#    esquema Thompson completo, cada uma de 2 momentos)
# =====================================================================
zeta = np.zeros((nx, nz))
psi = np.zeros((nx, nz))
thp = np.zeros((nx, nz))
qvp = np.zeros((nx, nz))
qc = np.zeros((nx, nz)); Nc = np.zeros((nx, nz))
qr = np.zeros((nx, nz)); Nr = np.zeros((nx, nz))
qi = np.zeros((nx, nz)); Ni = np.zeros((nx, nz))
qs = np.zeros((nx, nz)); Ns = np.zeros((nx, nz))
qg = np.zeros((nx, nz)); Ng = np.zeros((nx, nz))

surface_rain_accum = np.zeros(nx)

X0, Z0 = 4500.0, 500.0
RX, RZ = 1100.0, 600.0
r2 = ((X-X0)/RX)**2 + ((Z-Z0)/RZ)**2
if not CICLO_DIURNO:
    thp += DTHETA_BUBBLE*np.exp(-r2)*(r2 < 6)
    qvp += 0.5e-3*np.exp(-r2)*(r2 < 6)

def dispara_termica(t_now, h_now, amp_scale):
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
# 4. OPERADORES NUMERICOS (nucleo dinamico -- Equacao 2.6, inalterado)
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

# =====================================================================
# 4b. VELOCIDADES TERMINAIS VETORIZADAS (formulas algebricas puras --
#     vetorizado com NumPy sobre toda a grade, ao contrario das REACOES
#     de microfisica, que usam loop Python coluna a coluna. Mesmas
#     formulas/constantes de microfisica/processos_*.py.)
# =====================================================================
def _campo_Vt(q_f, N_f, rho_f, rho_x, mu, a, b, Vmax, correcao_rho=None):
    valido = (q_f > QMIN) & (N_f > NMIN)
    q_safe = np.maximum(q_f, QMIN)
    N_safe = np.maximum(N_f, NMIN)
    lam = (np.pi*rho_x*gamma_func(mu+4.0)*N_safe /
           (6.0*rho_f*gamma_func(mu+1.0)*q_safe)) ** (1.0/3.0)
    Vq = a*(gamma_func(mu+4.0+b)/gamma_func(mu+4.0))*lam**(-b)
    Vn = a*(gamma_func(mu+1.0+b)/gamma_func(mu+1.0))*lam**(-b)
    if correcao_rho is not None:
        Vq = Vq*correcao_rho
        Vn = Vn*correcao_rho
    Vq = np.where(valido, np.minimum(Vq, Vmax), 0.0)
    Vn = np.where(valido, np.minimum(Vn, Vmax), 0.0)
    return Vq, Vn

def campo_Vt_chuva(q_f, N_f, rho_f):
    correcao = (1.2/rho_f)**0.5
    return _campo_Vt(q_f, N_f, rho_f, rho_w, MU_RAIN, 842.0, 0.8, 9.5, correcao)

def campo_Vt_gelo(q_f, N_f, rho_f):
    return _campo_Vt(q_f, N_f, rho_f, rho_i, MU_ICE, 700.0, 1.0, 1.5)

def campo_Vt_neve(q_f, N_f, rho_f):
    return _campo_Vt(q_f, N_f, rho_f, rho_s, MU_SNOW, 11.72, 0.41, 3.0)

def campo_Vt_graupel(q_f, N_f, rho_f):
    return _campo_Vt(q_f, N_f, rho_f, rho_g, MU_SNOW, 19.3, 0.37, 12.0)

# =====================================================================
# 5. LOOP TEMPORAL
# =====================================================================
dt = 1.5
nsteps = int(T_END/dt)
save_every = int(300/dt)

frames_qc, frames_qr, frames_w, frames_thp, frames_t = [], [], [], [], []
frames_qvp, frames_u = [], []
frames_h_clc, frames_SHF = [], []

import time as _time
_t_wall_start = _time.time()

for step in range(nsteps+1):
    u, w = velocities(psi)
    t_now = step*dt

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

    if RADIACAO:
        THETA_ENV_now = theta_env_now_1d[None, :] - RAD_COOL_RATE*t_now
    else:
        THETA_ENV_now = theta_env_now_1d[None, :]

    T = (THETA_ENV_now+thp)*PI[None, :]

    # =================================================================
    # MICROFISICA: esquema Thompson completo (Passo 3), coluna a coluna
    # =================================================================
    if MICROFISICA == "thompson":
        qv_total = qv_env_now_1d[None, :] + qvp

        for i in range(nx):
            (T_novo, qv_novo, qc_novo, Nc_novo, qr_novo, Nr_novo,
             qi_novo, Ni_novo, qs_novo, Ns_novo, qg_novo, Ng_novo) = passo_microfisica_coluna(
                dt, T[i, :], P_PA_1d, RHO0_1d, qv_total[i, :],
                qc[i, :], Nc[i, :], qr[i, :], Nr[i, :],
                qi[i, :], Ni[i, :], qs[i, :], Ns[i, :], qg[i, :], Ng[i, :],
                evap_chuva=EVAP_CHUVA,
            )
            # devolve as variaveis do modelo dinamico (perturbacoes) --
            # o ambiente (theta_env, qv_env) e mantido fixo durante o
            # sub-passo de microfisica, entao toda a mudanca vira
            # perturbacao (thp, qvp), exatamente como no ajuste de
            # saturacao original
            thp[i, :] += (T_novo - T[i, :]) / PI
            qvp[i, :] += (qv_novo - qv_total[i, :])
            qc[i, :], Nc[i, :] = qc_novo, Nc_novo
            qr[i, :], Nr[i, :] = qr_novo, Nr_novo
            qi[i, :], Ni[i, :] = qi_novo, Ni_novo
            qs[i, :], Ns[i, :] = qs_novo, Ns_novo
            qg[i, :], Ng[i, :] = qg_novo, Ng_novo

    if step == nsteps:
        break

    # --- velocidades terminais de sedimentacao (vetorizado) ---
    if MICROFISICA == "thompson":
        Vtq_r, Vtn_r = campo_Vt_chuva(qr, Nr, RHO0_2d)
        Vtq_i, Vtn_i = campo_Vt_gelo(qi, Ni, RHO0_2d)
        Vtq_s, Vtn_s = campo_Vt_neve(qs, Ns, RHO0_2d)
        Vtq_g, Vtn_g = campo_Vt_graupel(qg, Ng, RHO0_2d)
    else:
        Vtq_r = Vtn_r = Vtq_i = Vtn_i = Vtq_s = Vtn_s = Vtq_g = Vtn_g = 0.0

    # --- tendencias dinamicas ---
    thv = thp + 0.61*theta0*qvp - theta0*(qc+qi+qr+qs+qg)
    dthv_dx = np.zeros_like(thp)
    dthv_dx[1:-1, :] = (thv[2:, :]-thv[:-2, :])/(2*dx)
    buoy_torque = (g/theta0)*dthv_dx

    dzeta = upwind_advect(zeta, u, w) + buoy_torque + K_DIFF*laplacian(zeta)
    dthp = upwind_advect(thp, u, w) - w*dtheta_dz_now_1d[None, :] + K_DIFF*laplacian(thp)
    dqvp = upwind_advect(qvp, u, w) - w*dqv_dz_now_1d[None, :] + K_DIFF*laplacian(qvp)
    dqc = upwind_advect(qc, u, w) + K_DIFF*laplacian(qc)
    dNc = upwind_advect(Nc, u, w) + K_DIFF*laplacian(Nc)

    if MICROFISICA == "thompson":
        dqr = upwind_advect(qr, u, w-Vtq_r) + K_DIFF*laplacian(qr)
        dNr = upwind_advect(Nr, u, w-Vtn_r) + K_DIFF*laplacian(Nr)
        dqi = upwind_advect(qi, u, w-Vtq_i) + K_DIFF*laplacian(qi)
        dNi = upwind_advect(Ni, u, w-Vtn_i) + K_DIFF*laplacian(Ni)
        dqs = upwind_advect(qs, u, w-Vtq_s) + K_DIFF*laplacian(qs)
        dNs = upwind_advect(Ns, u, w-Vtn_s) + K_DIFF*laplacian(Ns)
        dqg = upwind_advect(qg, u, w-Vtq_g) + K_DIFF*laplacian(qg)
        dNg = upwind_advect(Ng, u, w-Vtn_g) + K_DIFF*laplacian(Ng)
    else:
        dqr = dNr = dqi = dNi = dqs = dNs = dqg = dNg = 0.0

    zeta = zeta + dt*dzeta
    thp = thp + dt*dthp
    qvp = qvp + dt*dqvp
    qc = np.maximum(qc + dt*dqc, 0.0)
    Nc = np.maximum(Nc + dt*dNc, 0.0)
    if MICROFISICA == "thompson":
        qr = np.maximum(qr + dt*dqr, 0.0); Nr = np.maximum(Nr + dt*dNr, 0.0)
        qi = np.maximum(qi + dt*dqi, 0.0); Ni = np.maximum(Ni + dt*dNi, 0.0)
        qs = np.maximum(qs + dt*dqs, 0.0); Ns = np.maximum(Ns + dt*dNs, 0.0)
        qg = np.maximum(qg + dt*dqg, 0.0); Ng = np.maximum(Ng + dt*dNg, 0.0)

    zeta = apply_bc(zeta, zero_grad=False)
    thp = apply_bc(thp, zero_grad=True)
    qvp = apply_bc(qvp, zero_grad=True)
    qc = apply_bc(qc, zero_grad=True); Nc = apply_bc(Nc, zero_grad=True)
    if MICROFISICA == "thompson":
        qr = apply_bc(qr, zero_grad=True); Nr = apply_bc(Nr, zero_grad=True)
        qi = apply_bc(qi, zero_grad=True); Ni = apply_bc(Ni, zero_grad=True)
        qs = apply_bc(qs, zero_grad=True); Ns = apply_bc(Ns, zero_grad=True)
        qg = apply_bc(qg, zero_grad=True); Ng = apply_bc(Ng, zero_grad=True)

    psi = poisson_jacobi(zeta, psi, niter=120)

    if step % save_every == 0:
        frames_qc.append((qc+qi).copy())
        frames_qr.append((qr+qs+qg).copy())
        frames_w.append(w.copy())
        frames_thp.append(thp.copy())
        frames_qvp.append(qvp.copy())
        frames_u.append(u.copy())
        frames_t.append(step*dt)
        cloud_top = z[np.where((qc+qi).max(axis=0) > 1e-5)[0].max()] if (qc+qi).max() > 1e-5 else 0
        extra = f" | hora={6+t_now/3600:5.2f}h | h_CLC={h_clc:5.0f}m | SHF={SHF(t_now):5.0f}W/m2" if CICLO_DIURNO else ""
        wall_min = (_time.time()-_t_wall_start)/60.0
        print(f"t={step*dt/60:5.1f} min | w_max={w.max():5.2f} m/s | w_min={w.min():5.2f} m/s | "
              f"qc_max={(qc+qi).max()*1000:5.3f} g/kg | qg_max={qg.max()*1000:5.4f} g/kg | "
              f"topo_nuvem~{cloud_top:6.0f} m | thp_min={thp.min():5.2f} K"
              f"{extra} | tempo_real={wall_min:5.1f} min")

print("Simulacao concluida.")

# =====================================================================
# 6. FIGURAS
# =====================================================================
frame_dt_days = (frames_t[1]-frames_t[0])/86400.0 if len(frames_t) > 1 else 1.0
frames_dtheta_dt, frames_dq_dt, frames_du_dt = [], [], []
for idx in range(len(frames_t)):
    if idx == 0:
        frames_dtheta_dt.append(np.zeros_like(frames_thp[0]))
        frames_dq_dt.append(np.zeros_like(frames_qvp[0]))
        frames_du_dt.append(np.zeros_like(frames_u[0]))
    else:
        frames_dtheta_dt.append((frames_thp[idx]-frames_thp[idx-1])/frame_dt_days)
        frames_dq_dt.append((frames_qvp[idx]-frames_qvp[idx-1])*1000.0/frame_dt_days)
        frames_du_dt.append((frames_u[idx]-frames_u[idx-1])/frame_dt_days*86400.0)

n_frames = min(6, len(frames_t))
idxs = np.linspace(0, len(frames_t)-1, n_frames).astype(int)

fig, axes = plt.subplots(6, n_frames, figsize=(3.0*n_frames, 15), sharex=True, sharey=True)
if n_frames == 1:
    axes = axes.reshape(6, 1)

for j, idx in enumerate(idxs):
    t_min = frames_t[idx]/60.0

    im0 = axes[0, j].contourf(x/1000, z/1000, (frames_qc[idx]*1000).T, levels=20, cmap="Blues")
    axes[0, j].set_title(f"t={t_min:.0f} min\nqc+qi [g/kg]", fontsize=8)

    im1 = axes[1, j].contourf(x/1000, z/1000, frames_w[idx].T, levels=20, cmap="RdBu_r")
    axes[1, j].set_title("w [m/s]", fontsize=8)

    im2 = axes[2, j].contourf(x/1000, z/1000, (frames_qr[idx]*1000).T, levels=20, cmap="Greens")
    axes[2, j].set_title("chuva+neve+graupel [g/kg]", fontsize=8)

    im3 = axes[3, j].contourf(x/1000, z/1000, frames_dtheta_dt[idx].T, levels=20, cmap="RdBu_r")
    axes[3, j].set_title("dtheta/dt [K/dia]", fontsize=8)

    im4 = axes[4, j].contourf(x/1000, z/1000, frames_dq_dt[idx].T, levels=20, cmap="BrBG")
    axes[4, j].set_title("dq/dt [g/kg/dia]", fontsize=8)

    im5 = axes[5, j].contourf(x/1000, z/1000, frames_du_dt[idx].T, levels=20, cmap="PuOr")
    axes[5, j].set_title("du/dt [m/s/dia]", fontsize=8)

    if j == 0:
        for row in range(6):
            axes[row, j].set_ylabel("altura [km]", fontsize=8)
    axes[5, j].set_xlabel("x [km]", fontsize=8)

fig.suptitle(f"Modelo 2D + esquema Thompson completo -- {CENARIO} (bolha={DTHETA_BUBBLE}K, "
             f"microfisica={MICROFISICA}, evap_chuva={EVAP_CHUVA}, radiacao={RADIACAO})", fontsize=10)
plt.tight_layout()
plt.savefig(f"./nuvem_2d_thompson_evolucao_{CENARIO}.png", dpi=120)
plt.close()

# --- diagnostico de cold pool ---
thp_min_sfc = [frames_thp[idx][:, 0:3].min() for idx in range(len(frames_t))]
plt.figure(figsize=(7, 4))
plt.plot(np.array(frames_t)/60.0, thp_min_sfc, marker="o")
plt.axhline(0, color="gray", ls="--", lw=0.8)
plt.xlabel("Tempo (min)")
plt.ylabel("theta' minimo perto do chao (K)")
plt.title(f"Cold pool -- {CENARIO}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"./nuvem_2d_thompson_coldpool_{CENARIO}.png", dpi=120)
plt.close()

print("\nFiguras salvas:")
print(f" - ./nuvem_2d_thompson_evolucao_{CENARIO}.png")
print(f" - ./nuvem_2d_thompson_coldpool_{CENARIO}.png")

# --- dados numericos ---
np.savez_compressed(
    f"./resultados_nuvem_2d_thompson_{CENARIO}.npz",
    t_s=np.array(frames_t), x_m=x, z_m=z,
    qc=np.array(frames_qc), qr=np.array(frames_qr), w=np.array(frames_w),
    thp=np.array(frames_thp), qvp=np.array(frames_qvp), u=np.array(frames_u),
)
print(f" - ./resultados_nuvem_2d_thompson_{CENARIO}.npz")
