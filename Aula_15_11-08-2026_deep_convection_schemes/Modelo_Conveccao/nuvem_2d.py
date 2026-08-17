# -*- coding: utf-8 -*-
"""
MODELO 2D DE NUVEM CONVECTIVA (vorticidade-funcao de corrente, Boussinesq)
============================================================================
Curso: Conveccao Atmosferica - Regimes de Transicao

OBJETIVO
--------
Ate aqui, o curso trabalhou com um modelo de COLUNA (1D): a pluma
entranhante integra propriedades medias ao longo de z, e o fluxo de
massa M_c PARAMETRIZA o efeito da nuvem sobre o ambiente. Isso e
exatamente o que um esquema de cumulus faz dentro de um modelo de
previsao - porque rodar uma nuvem explicita em cada ponto de grade de
um GCM seria caro demais.

Este script faz o oposto: resolve a nuvem EXPLICITAMENTE em 2D (x,z),
sem parametrizar fluxo de massa nenhum. A "pluma" deixa de ser uma
abstracao (M_c, entranhamento eps) e passa a ser um campo de velocidade
vertical w(x,z,t) e um campo de agua de nuvem q_c(x,z,t) que se
desenvolvem sozinhos a partir das equacoes de movimento - o mesmo
espirito de um CRM (Cloud-Resolving Model) bem simplificado.

Comparar os dois lado a lado em aula e o ponto pedagogico central: o
que o modelo de coluna CHAMA de "entranhamento eps" aqui aparece
naturalmente como a mistura turbulenta nas bordas da bolha ascendente
resolvida explicitamente.

FORMULACAO
----------
Aproximacao de Boussinesq 2D em vorticidade-funcao de corrente
(equacoes anelasticas simplificadas, densidade de referencia
constante - valida para uma camada nao muito profunda, aceitavel
para fins didaticos):

  u = -dpsi/dz ,  w = dpsi/dx                          (define a funcao de corrente psi)
  nabla^2psi = zeta                                            (Poisson; zeta = vorticidade)
  dzeta/dt = -udzeta/dx - wdzeta/dz + (g/theta_0)dtheta'_v/dx + Knabla^2zeta  (vorticidade: o termo de empuxo
                                                       horizontal GERA circulacao - e
                                                       o analogo dinamico do termo de
                                                       empuxo B do modelo de coluna)
  dtheta'/dt = -udtheta'/dx - wdtheta'/dz - w.dtheta_env/dz + (L/(cpPi)).C + Knabla^2theta'
  dq_v/dt = -udq_v/dx - wdq_v/dz - w.dq_venv/dz - C + Knabla^2q_v
  dq_c/dt = -udq_c/dx - wdq_c/dz + C - autoconversao + Knabla^2q_c

onde C e a taxa de condensacao (ajuste de saturacao instantaneo a cada
passo) e K e uma DIFUSAO NUMERICA que tambem funciona, neste modelo
simplificado, como um proxy grosseiro da mistura turbulenta de
subgrade (o "entranhamento" resolvido implicitamente pela propria
dinamica, em vez de parametrizado).

SIMPLIFICACOES (deixadas explicitas)
-------------------------------------
  - Adveccao upwind de 1a ordem (robusta, mas difusiva - modelos reais
    usam esquemas de ordem mais alta);
  - Solver de Poisson por iteracao de Jacobi (simples e vetorizado, nao
    o mais rapido possivel, mas facil de verificar);
  - Sem microfisica de gelo, sem downdraft explicito, sem queda de
    precipitacao com velocidade terminal - apenas remocao simples de
    q_c acima de um limiar (chuva "desaparece" ao ser formada, nao e
    transportada);
  - Paredes rigidas nas quatro bordas do dominio (psi=0), sem ciclo
    diurno nem heterogeneidade de superficie - so o desenvolvimento de
    UMA bolha termica inicial.
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
# 1. CONSTANTES E GRADE
# =====================================================================
g = 9.81
cp = 1004.0
Rd = 287.0
Rv = 461.5
Lv = 2.5e6
eps_R = Rd/Rv
p0 = 1000.0
theta0 = 300.0     # temperatura potencial de referencia (Boussinesq) [K]

nx, nz = 90, 110
dx, dz = 100.0, 100.0      # 9 km x 11 km
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

# =====================================================================
# 2. ESTADO BASE (ambiente) - mesma filosofia do modelo de coluna:
#    camada quase-neutra -> capa estavel (CIN) -> camada fracamente
#    estavel em theta mas condicionalmente instavel na saturacao (CAPE)
#    -> capa estavel no topo (tropopausa efetiva do dominio)
# =====================================================================
def dtheta_dz_env(zz):
    return np.where(zz < 1000, 3.0e-3,
           np.where(zz < 2000, 6.5e-3,      # capa estavel -> CIN
           np.where(zz < 8500, 2.0e-3,      # fracamente estavel em theta, mas
                                              # instavel na saturacao (CAPE via
                                              # calor latente)
                    6.0e-3)))                # capa estavel no topo

theta_env_1d = np.zeros(nz)
theta_env_1d[0] = theta0
for k in range(1, nz):
    theta_env_1d[k] = theta_env_1d[k-1] + dtheta_dz_env(z[k-1])*dz

theta_env_1d = np.zeros(nz)
theta_env_1d[0] = theta0
for k in range(1, nz):
    theta_env_1d[k] = theta_env_1d[k-1] + dtheta_dz_env(z[k-1])*dz

PI = exner(z)
P = p_of_z(z)
T_env_1d = theta_env_1d*PI
QSAT_ENV_1d = qsat_liq(T_env_1d, P)

def RH_env_profile(zz):
    # umidade relativa de fundo do ambiente -- SEMPRE subsaturada, para que
    # a nuvem so apareca onde a bolha realmente levantar e resfriar o ar
    # ate a saturacao (nao o ambiente "de fundo" sozinho)
    return np.where(zz < 1000, 0.70,
           np.where(zz < 2000, 0.35,     # capa seca -> reforca o CIN
           np.where(zz < 8500, 0.55,
                    0.20)))

RH_env_1d = RH_env_profile(z)
qv_env_1d = RH_env_1d*QSAT_ENV_1d

THETA_ENV = np.broadcast_to(theta_env_1d[None, :], (nx, nz)).copy()
QV_ENV = np.broadcast_to(qv_env_1d[None, :], (nx, nz)).copy()
DTHETA_ENV_DZ = np.gradient(theta_env_1d, dz)
DQV_ENV_DZ = np.gradient(qv_env_1d, dz)

# =====================================================================
# 3. CAMPOS PROGNOSTICOS
# =====================================================================
zeta = np.zeros((nx, nz))
psi  = np.zeros((nx, nz))
thp  = np.zeros((nx, nz))   # theta' (perturbacao)
qvp  = np.zeros((nx, nz))   # qv' (perturbacao)
qc   = np.zeros((nx, nz))   # agua de nuvem (nao-negativa)
rain_removed = np.zeros((nx, nz))  # acumulado de "chuva" removida (diagnostico)


# --- Bolha termica inicial (o gatilho) ---
X0, Z0 = 4500.0, 500.0
RX, RZ = 1100.0, 600.0
DTHETA_BUBBLE = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0   # K
CENARIO = sys.argv[2] if len(sys.argv) > 2 else "bolha"
r2 = ((X-X0)/RX)**2 + ((Z-Z0)/RZ)**2
thp += DTHETA_BUBBLE*np.exp(-r2)*(r2 < 6)
qvp += 0.5e-3*np.exp(-r2)*(r2 < 6)   # bolha tambem um pouco mais umida que o ambiente local

# =====================================================================
# 4. OPERADORES NUMERICOS
# =====================================================================
def poisson_jacobi(zeta_field, psi_guess, niter=200, omega=1.0):
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

K_DIFF = 25.0        # m^2/s -- difusao numerica (proxy grosseiro de mistura turbulenta)
QC_CRIT = 1.0e-3      # kg/kg -- limiar de autoconversao (chuva "some" acima disso)
TAU_AUTO = 400.0      # s -- escala de tempo da autoconversao

# =====================================================================
# 5. LOOP TEMPORAL
# =====================================================================
dt = 1.5
t_end = 3600.0
nsteps = int(t_end/dt)
save_every = int(300/dt)   # salva um quadro a cada 5 minutos simulados

frames_qc, frames_w, frames_thp, frames_t = [], [], [], []

for step in range(nsteps+1):
    u, w = velocities(psi)

    # --- ajuste de saturacao (condensacao/evaporacao instantanea) ---
    T = (THETA_ENV+thp)*PI[None, :]
    qs = qsat_liq(T, P[None, :])
    qv_total = QV_ENV+qvp
    cond = qv_total - qs   # >0 condensa, <0 pode evaporar q_c

    to_condense = np.maximum(cond, 0.0)
    to_evaporate = np.minimum(np.maximum(-cond, 0.0), qc)

    qvp += -to_condense + to_evaporate
    qc  += to_condense - to_evaporate
    thp += (Lv/(cp*PI[None, :]))*(to_condense-to_evaporate)

    # --- autoconversao simples (remove q_c acima do limiar) ---
    excess = np.maximum(qc-QC_CRIT, 0.0)
    removed = excess*(1-np.exp(-dt/TAU_AUTO))
    qc -= removed
    rain_removed += removed

    if step == nsteps:
        break

    # --- tendencias dinamicas ---
    dthv_dx = np.zeros_like(thp)
    thv = thp + 0.61*theta0*qvp   # aprox. de temperatura potencial virtual perturbada
    dthv_dx[1:-1, :] = (thv[2:, :]-thv[:-2, :])/(2*dx)
    buoy_torque = (g/theta0)*dthv_dx

    dzeta = upwind_advect(zeta, u, w) + buoy_torque + K_DIFF*laplacian(zeta)
    dthp  = upwind_advect(thp, u, w) - w*DTHETA_ENV_DZ[None, :] + K_DIFF*laplacian(thp)
    dqvp  = upwind_advect(qvp, u, w) - w*DQV_ENV_DZ[None, :] + K_DIFF*laplacian(qvp)
    dqc   = upwind_advect(qc, u, w) + K_DIFF*laplacian(qc)

    zeta = zeta + dt*dzeta
    thp  = thp  + dt*dthp
    qvp  = qvp  + dt*dqvp
    qc   = np.maximum(qc + dt*dqc, 0.0)

    zeta = apply_bc(zeta, zero_grad=False)
    thp  = apply_bc(thp,  zero_grad=True)
    qvp  = apply_bc(qvp,  zero_grad=True)
    qc   = apply_bc(qc,   zero_grad=True)

    psi = poisson_jacobi(zeta, psi, niter=120)

    if step % save_every == 0:
        frames_qc.append(qc.copy())
        frames_w.append(w.copy())
        frames_thp.append(thp.copy())
        frames_t.append(step*dt)
        wmax = w.max()
        print(f"t={step*dt/60:5.1f} min | w_max={wmax:5.2f} m/s | "
              f"qc_max={qc.max()*1000:5.3f} g/kg | topo_nuvem~"
              f"{(z[np.where(qc.max(axis=0)>1e-5)[0].max()] if (qc>1e-5).any() else 0):6.0f} m")

print("Simulacao concluida.")

# =====================================================================
# 6. FIGURA: evolucao da nuvem em varios horarios
# =====================================================================
n_panels = min(6, len(frames_qc))
idxs = np.linspace(0, len(frames_qc)-1, n_panels).astype(int)

fig, axs = plt.subplots(2, n_panels, figsize=(3.1*n_panels, 7), sharex=True, sharey=True)
for j, idx in enumerate(idxs):
    t_min = frames_t[idx]/60
    ax1 = axs[0, j]
    pc1 = ax1.contourf(X/1000, Z/1000, frames_qc[idx]*1000, levels=np.linspace(0, 2, 11),
                        cmap="Blues", extend="max")
    ax1.set_title(f"t={t_min:.0f} min\nq_c [g/kg]", fontsize=9)
    if j == 0: ax1.set_ylabel("altura [km]")

    ax2 = axs[1, j]
    lim = max(1.0, np.abs(frames_w[idx]).max())
    pc2 = ax2.contourf(X/1000, Z/1000, frames_w[idx], levels=np.linspace(-lim, lim, 15),
                        cmap="RdBu_r")
    ax2.set_title("w [m/s]", fontsize=9)
    ax2.set_xlabel("x [km]")
    if j == 0: ax2.set_ylabel("altura [km]")

fig.suptitle(f"Modelo 2D explicito de nuvem convectiva - cenario: {CENARIO} (Deltatheta={DTHETA_BUBBLE}K)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"nuvem_2d_evolucao_{CENARIO}.png", dpi=150)
print(f"Figura salva: nuvem_2d_evolucao_{CENARIO}.png")

# --- Figura extra: snapshot unico mais detalhado no horario de maior desenvolvimento ---
idx_max = int(np.argmax([f.max() for f in frames_qc]))
fig2, ax = plt.subplots(figsize=(6, 7))
cf = ax.contourf(X/1000, Z/1000, frames_qc[idx_max]*1000, levels=np.linspace(0, 2, 11),
                  cmap="Blues", extend="max")
ax.contour(X/1000, Z/1000, frames_w[idx_max], levels=[1, 3, 5, 8], colors="black", linewidths=0.8)
plt.colorbar(cf, ax=ax, label="q_c [g/kg]")
ax.set_xlabel("x [km]"); ax.set_ylabel("altura [km]")
ax.set_title(f"Nuvem em t={frames_t[idx_max]/60:.0f} min - {CENARIO}\n(contornos pretos: w em m/s)")
plt.tight_layout()
plt.savefig(f"nuvem_2d_detalhe_{CENARIO}.png", dpi=150)
print(f"Figura salva: nuvem_2d_detalhe_{CENARIO}.png")
