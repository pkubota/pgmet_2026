"""
==============================================================================
COMPARACAO CONSOLIDADA - TODOS OS NIVEIS, CICLO DIURNO COMPLETO
Disciplina: Modelagem de Camada Limite
==============================================================================
Roda os 5 estagios da disciplina (Ekman puro -> MYNN com TKE prognostica)
usando EXATAMENTE a mesma grade vertical, o mesmo vento geostrofico, o
mesmo ciclo diurno de fluxo de calor de superficie, e o mesmo solver de
difusao implicita -- variando APENAS a fisica do fechamento de turbulencia.

ESTAGIOS:
  1) Ekman puro           - K constante (Nivel 1)
  2) O'Brien nao-local    - K(z) convectivo prescrito; NOTURNO = K_bg (Nivel 2)
  3) MY82 diagnostico     - fechamento local, Ri_crit real ~0.03 (Nivel 3)
  4) MYNN diagnostico     - fechamento local, Ri_crit real ~0.95 (Nivel 3b)
  5) MYNN TKE prognostica - q^2 com memoria temporal (Nivel 4)

DURACAO: parametrizavel (24h ou 72h). Ciclo diurno de 24h se repete.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
import time as timer

g = 9.81
k_von = 0.4

# ===========================================================================
# CONFIGURACAO COMUM A TODOS OS ESTAGIOS
# ===========================================================================
N_HORAS   = 72.0          # <-- mude para 24.0 se quiser rodar so' 1 dia
H_dom     = 2000.0
dz        = 50.0
dt        = 60.0
n_steps   = int(N_HORAS*3600/dt)

f_cor     = 1.0e-4        # ~43 graus de latitude (periodo inercial ~17.4h)
ug, vg    = 8.0, 0.0
theta0_ini = 290.0
gamma_free = 4.0e-3       # K/m, lapse rate da atmosfera livre acima da CLP inicial
h0_ini     = 100.0

z  = np.arange(0.0, H_dom+dz, dz)
nz = len(z)


def fluxo_calor_diurno(t_sec, Q0max=0.20, Q0min=-0.03):
    """Ciclo diurno de 24h, repetido continuamente (para N_HORAS>24)."""
    hora = (t_sec/3600.0) % 24.0
    if 6.0 <= hora <= 18.0:
        return Q0max*np.sin(np.pi*(hora-6.0)/12.0)
    return Q0min


# ---------------------------------------------------------------------------
# Solver tridiagonal comum
# ---------------------------------------------------------------------------
def thomas_solver(a, b, c, d):
    n = len(d)
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = c[0]/b[0]; dp[0] = d[0]/b[0]
    for i in range(1, n):
        m = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/m
        dp[i] = (d[i]-a[i]*dp[i-1])/m
    x = np.zeros(n); x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x


def difusao_implicita_Kvar(phi, K, dz, dt, flux_surf=None, phi_surf=None):
    n = len(phi)
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n); d = np.zeros(n)
    K_face = 0.5*(K[:-1]+K[1:])
    for kk in range(1, n-1):
        rp = K_face[kk]*dt/dz**2
        rm = K_face[kk-1]*dt/dz**2
        a[kk] = -rm; b[kk] = 1.0+rp+rm; c[kk] = -rp; d[kk] = phi[kk]
    if phi_surf is not None:
        b[0] = 1.0; c[0] = 0.0; d[0] = phi_surf
    else:
        rp = K_face[0]*dt/dz**2
        b[0] = 1.0+rp; c[0] = -rp; d[0] = phi[0] + dt*flux_surf/dz
    rm = K_face[-1]*dt/dz**2
    a[-1] = -rm; b[-1] = 1.0+rm; d[-1] = phi[-1]
    return thomas_solver(a, b, c, d)


def diagnostica_h(u, v, theta, z, Ric=0.25, theta_va=290.0, h_min=50.0):
    """Diagnostico comum de altura da CLP (bulk Ri) p/ TODOS os estagios,
    usado so' como metrica de COMPARACAO (nao afeta a fisica interna)."""
    theta_s = theta[0]
    for kk in range(1, len(z)):
        Usq = max(u[kk]**2+v[kk]**2, 1e-4)
        Rib = g*(theta[kk]-theta_s)*z[kk]/(theta_va*Usq)
        if Rib >= Ric:
            Usq_p = max(u[kk-1]**2+v[kk-1]**2, 1e-4)
            Rib_p = g*(theta[kk-1]-theta_s)*z[kk-1]/(theta_va*Usq_p)
            frac = (Ric-Rib_p)/(Rib-Rib_p+1e-12)
            return max(z[kk-1]+frac*(z[kk]-z[kk-1]), h_min)
    return z[-1]


# ===========================================================================
# ESTAGIO 1: EKMAN PURO (K constante)
# ===========================================================================
def estagio1_ekman(K_const=5.0):
    u = np.full(nz, ug); v = np.full(nz, vg); u[0], v[0] = 0.0, 0.0
    theta = theta0_ini + np.where(z <= h0_ini, 0.0, gamma_free*(z-h0_ini))
    K = np.full(nz, K_const)

    h_hist, t_hist = [], []
    theta_hov = []
    t = 0.0
    for n in range(n_steps):
        Q0 = fluxo_calor_diurno(t)
        if n % 30 == 0:
            h_hist.append(diagnostica_h(u, v, theta, z, theta_va=theta0_ini))
            t_hist.append(t/3600.0)
            theta_hov.append(theta.copy())
        u_cor = u + dt*f_cor*(v-vg)
        v_cor = v - dt*f_cor*(u-ug)
        u = difusao_implicita_Kvar(u_cor, K, dz, dt, phi_surf=0.0)
        v = difusao_implicita_Kvar(v_cor, K, dz, dt, phi_surf=0.0)
        theta = difusao_implicita_Kvar(theta, K, dz, dt, flux_surf=Q0)
        t += dt
    return dict(nome="1: Ekman puro (K=cte)", t=np.array(t_hist), h=np.array(h_hist),
                theta_hov=np.array(theta_hov))


# ===========================================================================
# ESTAGIO 2: O'BRIEN NAO-LOCAL (convectivo) + K_bg noturno
# ===========================================================================
def perfil_K_obrien(z, h, u_star, Q0, theta_va, p=2.0, K_bg=1.0):
    if Q0 > 0:
        w_star_b = (g/theta_va*Q0*h)**(1.0/3.0)
        K = np.full_like(z, K_bg)
        mask = z <= h
        zc = z[mask]
        w_s = (u_star**3 + 8.0*k_von*w_star_b**3*(zc/max(h, 1.0)))**(1.0/3.0)
        K_conv = k_von*w_s*zc*(1.0-zc/h)**p
        K[mask] = np.maximum(K_conv, K_bg)
        return K
    else:
        # LIMITACAO CONHECIDA: o esquema O'Brien nao tem fechamento estavel
        # proprio -> a noite, cai no minimo de fundo (K_bg) em toda a coluna.
        return np.full_like(z, K_bg)


def estagio2_obrien(u_star=0.3, K_bg=1.0):
    u = np.full(nz, ug); v = np.full(nz, vg); u[0], v[0] = 0.0, 0.0
    theta = theta0_ini + np.where(z <= h0_ini, 0.0, gamma_free*(z-h0_ini))

    h_hist, t_hist, theta_hov = [], [], []
    t = 0.0
    for n in range(n_steps):
        Q0 = fluxo_calor_diurno(t)
        h_diag = diagnostica_h(u, v, theta, z, theta_va=theta0_ini)
        K = perfil_K_obrien(z, max(h_diag, 50.0), u_star, Q0, theta0_ini, K_bg=K_bg)

        if n % 30 == 0:
            h_hist.append(h_diag); t_hist.append(t/3600.0)
            theta_hov.append(theta.copy())

        u_cor = u + dt*f_cor*(v-vg)
        v_cor = v - dt*f_cor*(u-ug)
        u = difusao_implicita_Kvar(u_cor, K, dz, dt, phi_surf=0.0)
        v = difusao_implicita_Kvar(v_cor, K, dz, dt, phi_surf=0.0)
        theta = difusao_implicita_Kvar(theta, K, dz, dt, flux_surf=Q0)
        t += dt
    return dict(nome="2: O'Brien nao-local (+K_bg noturno)", t=np.array(t_hist),
                h=np.array(h_hist), theta_hov=np.array(theta_hov))


# ===========================================================================
# ESTAGIO 3: MY82 DIAGNOSTICO (Nivel 3)
# ===========================================================================
_A1,_B1,_A2,_B2,_C1 = 0.92, 16.6, 0.74, 10.1, 0.08
_g1_my = 1.0/3.0 - 2.0*_A1/_B1
_g2_my = _B2/_B1 + 6.0*_A1/_B1


def _SH_my(Rf):
    return 3.0*_A2*(_g1_my-(_g1_my+_g2_my)*Rf)/(1.0-Rf)


def _SM_my(Rf, SH):
    num = _B1*(_g1_my-_C1) - (_B1*(_g1_my-_C1)+6.0*(_A1+3.0*_A2))*Rf
    den = _B1*_g1_my - (_B1*(_g1_my+_g2_my)-3.0*_A1)*Rf
    return (_A1/_A2)*(num/den)*SH


def _Ri_de_Rf_my(Rf):
    SH = _SH_my(Rf); SM = _SM_my(Rf, SH)
    return (SM/SH)*Rf if abs(SH) > 1e-12 else np.nan


_res_my = minimize_scalar(lambda Rf: -_Ri_de_Rf_my(Rf), bounds=(-1.0, _g1_my/(_g1_my+_g2_my)-1e-6),
                           method='bounded')
_RF_PEAK_MY, _RI_CRIT_MY = _res_my.x, -_res_my.fun


def _resolve_Rf_my(Ri):
    lo, hi = -8.0, _RF_PEAK_MY
    def f(Rf):
        SH = _SH_my(Rf); SM = _SM_my(Rf, SH)
        return SM*Rf - Ri*SH
    try:
        if f(lo)*f(hi) > 0: return _RF_PEAK_MY, True
        return brentq(f, lo, hi, xtol=1e-10), False
    except Exception:
        return _RF_PEAK_MY, True


def calcula_K_MY82(dUdz, dVdz, dThetadz, z, theta0, lambda0=100.0, K_bg=0.3):
    n = len(z)
    S2 = np.maximum(dUdz**2+dVdz**2, 1e-8)
    Ri = (g/theta0)*dThetadz/S2
    Km = np.zeros(n); Kh = np.zeros(n)
    l = np.maximum(k_von*z/(1.0+k_von*z/lambda0), 1e-3)
    for k in range(n):
        Rf, col = _resolve_Rf_my(Ri[k])
        if col:
            Km[k] = K_bg; Kh[k] = K_bg; continue
        SH = _SH_my(Rf); SM = _SM_my(Rf, SH)
        q2 = _B1*l[k]**2*S2[k]*SM*(1.0-Rf)
        q = np.sqrt(max(q2, 0.0))
        Km[k] = max(l[k]*q*SM, K_bg)
        Kh[k] = max(l[k]*q*SH, K_bg)
    return Km, Kh


def estagio3_my82():
    u = np.full(nz, ug); v = np.full(nz, vg); u[0], v[0] = 0.0, 0.0
    theta = theta0_ini + np.where(z <= h0_ini, 0.0, gamma_free*(z-h0_ini))

    h_hist, t_hist, theta_hov = [], [], []
    t = 0.0
    for n in range(n_steps):
        Q0 = fluxo_calor_diurno(t)
        dUdz = np.gradient(u, z); dVdz = np.gradient(v, z); dThetadz = np.gradient(theta, z)
        Km, Kh = calcula_K_MY82(dUdz, dVdz, dThetadz, z, theta0_ini)

        if n % 30 == 0:
            h_hist.append(diagnostica_h(u, v, theta, z, theta_va=theta0_ini))
            t_hist.append(t/3600.0)
            theta_hov.append(theta.copy())

        u_cor = u + dt*f_cor*(v-vg)
        v_cor = v - dt*f_cor*(u-ug)
        u = difusao_implicita_Kvar(u_cor, Km, dz, dt, phi_surf=0.0)
        v = difusao_implicita_Kvar(v_cor, Km, dz, dt, phi_surf=0.0)
        theta = difusao_implicita_Kvar(theta, Kh, dz, dt, flux_surf=Q0)
        t += dt
    return dict(nome=f"3: MY82 diagnostico (Ri_crit={_RI_CRIT_MY:.3f})",
                t=np.array(t_hist), h=np.array(h_hist), theta_hov=np.array(theta_hov))


# ===========================================================================
# ESTAGIO 4: MYNN DIAGNOSTICO ("Nivel 2" do MYNN)
# ===========================================================================
_A1n,_A2n,_B1n,_B2n,_C1n = 1.18, 0.665, 24.0, 15.0, 0.137
_C2n,_C3n,_C5n = 0.75, 0.352, 0.2
_g1n = 0.235
_g2n = (2.0*_A1n*(3.0-2.0*_C2n) + _B2n*(1.0-_C3n))/_B1n
_F1n = _B1n*(_g1n-_C1n) + 2.0*_A1n*(3.0-2.0*_C2n) + 3.0*_A2n*(1.0-_C2n)*(1.0-_C5n)
_F2n = _B1n*(_g1n+_g2n) - 3.0*_A1n*(1.0-_C2n)
_Rf1n = _B1n*(_g1n-_C1n)/_F1n
_Rf2n = _B1n*_g1n/_F2n
_Rfc_n = _g1n/(_g1n+_g2n)


def _SH_nn(Rf):
    return 3.0*_A2n*(_g1n+_g2n)*(_Rfc_n-Rf)/(1.0-Rf)


def _SM_nn(Rf, SH):
    return (_A1n*_F1n)/(_A2n*_F2n)*(_Rf1n-Rf)/(_Rf2n-Rf)*SH


def _Ri_de_Rf_nn(Rf):
    SH = _SH_nn(Rf); SM = _SM_nn(Rf, SH)
    return (SM/SH)*Rf if abs(SH) > 1e-12 else np.nan


_hi_teto_nn = min(_Rfc_n, _Rf1n, _Rf2n) - 1e-6
_res_nn = minimize_scalar(lambda Rf: -_Ri_de_Rf_nn(Rf), bounds=(-8.0, _hi_teto_nn), method='bounded')
_RF_PEAK_NN, _RI_CRIT_NN = _res_nn.x, -_res_nn.fun


def _resolve_Rf_nn(Ri):
    lo, hi = -8.0, _RF_PEAK_NN
    def f(Rf):
        SH = _SH_nn(Rf); SM = _SM_nn(Rf, SH)
        return SM*Rf - Ri*SH
    try:
        if f(lo)*f(hi) > 0: return _RF_PEAK_NN, True
        return brentq(f, lo, hi, xtol=1e-10), False
    except Exception:
        return _RF_PEAK_NN, True


def calcula_K_MYNN_diag(dUdz, dVdz, dThetadz, z, theta0, lambda0=100.0, K_bg=0.3):
    n = len(z)
    S2 = np.maximum(dUdz**2+dVdz**2, 1e-8)
    Ri = (g/theta0)*dThetadz/S2
    Km = np.zeros(n); Kh = np.zeros(n)
    l = np.maximum(k_von*z/(1.0+k_von*z/lambda0), 1e-3)
    for k in range(n):
        Rf, col = _resolve_Rf_nn(Ri[k])
        if col:
            Km[k] = K_bg; Kh[k] = K_bg; continue
        SH = _SH_nn(Rf); SM = _SM_nn(Rf, SH)
        q2 = _B1n*l[k]**2*S2[k]*SM*(1.0-Rf)
        q = np.sqrt(max(q2, 0.0))
        Km[k] = max(l[k]*q*SM, K_bg)
        Kh[k] = max(l[k]*q*SH, K_bg)
    return Km, Kh


def estagio4_mynn_diag():
    u = np.full(nz, ug); v = np.full(nz, vg); u[0], v[0] = 0.0, 0.0
    theta = theta0_ini + np.where(z <= h0_ini, 0.0, gamma_free*(z-h0_ini))

    h_hist, t_hist, theta_hov = [], [], []
    t = 0.0
    for n in range(n_steps):
        Q0 = fluxo_calor_diurno(t)
        dUdz = np.gradient(u, z); dVdz = np.gradient(v, z); dThetadz = np.gradient(theta, z)
        Km, Kh = calcula_K_MYNN_diag(dUdz, dVdz, dThetadz, z, theta0_ini)

        if n % 30 == 0:
            h_hist.append(diagnostica_h(u, v, theta, z, theta_va=theta0_ini))
            t_hist.append(t/3600.0)
            theta_hov.append(theta.copy())

        u_cor = u + dt*f_cor*(v-vg)
        v_cor = v - dt*f_cor*(u-ug)
        u = difusao_implicita_Kvar(u_cor, Km, dz, dt, phi_surf=0.0)
        v = difusao_implicita_Kvar(v_cor, Km, dz, dt, phi_surf=0.0)
        theta = difusao_implicita_Kvar(theta, Kh, dz, dt, flux_surf=Q0)
        t += dt
    return dict(nome=f"4: MYNN diagnostico (Ri_crit={_RI_CRIT_NN:.2f})",
                t=np.array(t_hist), h=np.array(h_hist), theta_hov=np.array(theta_hov))


# ===========================================================================
# ESTAGIO 5: MYNN COM TKE PROGNOSTICA (Nivel 4)
# ===========================================================================
def _calcula_L(z, q, theta, dThetadz, u_star, L_MO, l_min=1.0):
    n = len(z)
    zeta = z/L_MO if np.isfinite(L_MO) and abs(L_MO) > 1e-6 else np.zeros(n)
    Ls = np.zeros(n)
    for k in range(n):
        zt = zeta[k]
        if zt >= 1.0: Ls[k] = k_von*z[k]/3.7
        elif zt >= 0.0: Ls[k] = k_von*z[k]/(1.0+2.7*zt)
        else: Ls[k] = k_von*z[k]*(1.0-100.0*zt)**0.2
    trapz_fn = getattr(np, 'trapezoid', None) or np.trapz
    Lt = max(0.23*trapz_fn(q*z, z)/max(trapz_fn(q, z), 1e-6), l_min)
    N2 = np.maximum((g/theta[0])*dThetadz, 0.0)
    N = np.sqrt(N2)
    Lb = np.where(N > 1e-8, q/np.maximum(N, 1e-8), 1e6)
    Ls = np.maximum(Ls, l_min*0.1)
    return np.maximum(1.0/(1.0/Ls + 1.0/Lt + 1.0/Lb), l_min)


def _SM25_SH25(GM, GH):
    F1 = 1.0 - 3.0*_A2n*_B2n*(1.0-_C3n)*GH
    F2 = 1.0 - 9.0*_A1n*_A2n*(1.0-_C2n)*GH
    F3 = F1 + 9.0*_A2n**2*(1.0-_C2n)*(1.0-_C5n)*GH
    F4 = F1 - 12.0*_A1n*_A2n*(1.0-_C2n)*GH
    F5 = 6.0*_A1n**2*GM
    D25 = F2*F4 + F5*F3
    D25 = np.where(np.abs(D25) < 1e-12, 1e-12, D25)
    SM25 = _A1n*(F3-3.0*_C1n*F4)/D25
    SH25 = _A2n*(F2+3.0*_C1n*F5)/D25
    return SM25, SH25


def estagio5_mynn_tke(K_bg=0.3, q_min=0.05):
    u = np.full(nz, ug); v = np.full(nz, vg); u[0], v[0] = 0.0, 0.0
    theta = theta0_ini + np.where(z <= h0_ini, 0.0, gamma_free*(z-h0_ini))
    q = np.full(nz, 0.3); q[0] = 0.01
    Km_prev = np.full(nz, 1.0); Kh_prev = np.full(nz, 1.0)

    h_hist, t_hist, theta_hov = [], [], []
    t = 0.0
    for n in range(n_steps):
        Q0 = fluxo_calor_diurno(t)
        dUdz = np.gradient(u, z); dVdz = np.gradient(v, z); dThetadz = np.gradient(theta, z)
        S2 = np.maximum(dUdz**2+dVdz**2, 1e-8)

        u_star2 = np.sqrt((Km_prev[0]*dUdz[0])**2+(Km_prev[0]*dVdz[0])**2)
        u_star = np.sqrt(max(u_star2, 1e-6))
        wtheta0 = Q0
        L_MO = -theta0_ini*u_star**3/(k_von*g*wtheta0) if abs(wtheta0) > 1e-10 else 1e6

        q_safe = np.maximum(q, q_min)
        L = _calcula_L(z, q_safe, theta, dThetadz, u_star, L_MO)
        GM = (L**2/q_safe**2)*S2
        GH = -(L**2/q_safe**2)*(g/theta0_ini)*dThetadz
        SM25, SH25 = _SM25_SH25(GM, GH)
        Km = np.maximum(L*q_safe*SM25, K_bg)
        Kh = np.maximum(L*q_safe*SH25, K_bg)
        Kq = 3.0*Km

        if n % 30 == 0:
            h_hist.append(diagnostica_h(u, v, theta, z, theta_va=theta0_ini))
            t_hist.append(t/3600.0)
            theta_hov.append(theta.copy())

        u_cor = u + dt*f_cor*(v-vg)
        v_cor = v - dt*f_cor*(u-ug)
        u = difusao_implicita_Kvar(u_cor, Km, dz, dt, phi_surf=0.0)
        v = difusao_implicita_Kvar(v_cor, Km, dz, dt, phi_surf=0.0)
        theta = difusao_implicita_Kvar(theta, Kh, dz, dt, flux_surf=Q0)

        Prod = 2.0*Km*S2 - 2.0*Kh*(g/theta0_ini)*dThetadz
        q2 = q_safe**2
        q2_prod = np.maximum(q2 + dt*Prod, 0.0)
        q2_diss = q2_prod/(1.0 + dt*2.0*q_safe/(_B1n*L))
        q2_diss = np.maximum(q2_diss, q_min**2)
        q2_diss[0] = q_min**2
        q2_new = np.maximum(difusao_implicita_Kvar(q2_diss, Kq, dz, dt, phi_surf=q_min**2), q_min**2)
        q = np.sqrt(q2_new)

        Km_prev, Kh_prev = Km, Kh
        t += dt
    return dict(nome="5: MYNN + TKE prognostica", t=np.array(t_hist),
                h=np.array(h_hist), theta_hov=np.array(theta_hov))


def difusao_implicita_com_fluxo_extra(phi, K, dz, dt, flux_extra_face=None,
                                       flux_surf=None, phi_surf=None):
    """
    Generalizacao de difusao_implicita_Kvar: alem da difusao implicita
    padrao, aceita um fluxo adicional CONHECIDO em cada interface
    (flux_extra_face, tamanho n-1), que entra explicitamente no lado
    direito -- exatamente o mecanismo usado para o termo de contra-
    gradiente (Kc*gamma_c) e o termo de entranhamento no YSU (eq. B1
    do paper original: dC/dt = d/dz[Kc(dC/dz - gamma_c)] - d/dz[(w'c')_h (z/h)^3]).
    """
    n = len(phi)
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n); d = np.zeros(n)
    K_face = 0.5*(K[:-1]+K[1:])
    if flux_extra_face is None:
        flux_extra_face = np.zeros(n-1)

    for kk in range(1, n-1):
        rp = K_face[kk]*dt/dz**2
        rm = K_face[kk-1]*dt/dz**2
        a[kk] = -rm; b[kk] = 1.0+rp+rm; c[kk] = -rp
        d[kk] = phi[kk] + dt/dz*(flux_extra_face[kk]-flux_extra_face[kk-1])

    if phi_surf is not None:
        b[0] = 1.0; c[0] = 0.0; d[0] = phi_surf
    else:
        rp = K_face[0]*dt/dz**2
        b[0] = 1.0+rp; c[0] = -rp
        fs = flux_surf if flux_surf is not None else 0.0
        d[0] = phi[0] + dt*fs/dz + dt/dz*flux_extra_face[0]

    rm = K_face[-1]*dt/dz**2
    a[-1] = -rm; b[-1] = 1.0+rm
    d[-1] = phi[-1] - dt/dz*flux_extra_face[-1]
    return thomas_solver(a, b, c, d)


# ===========================================================================
# ESTAGIO 6: YSU COMPLETO (Hong, Noh & Dudhia 2006)
# ===========================================================================
# Mixed layer (z<=h): Km=k*ws*z*(1-z/h)^2, contra-gradiente gamma_c (so' para
# theta, por simplicidade -- o paper original aplica tambem a u,v), termo de
# entranhamento assintotico (w'theta')_h*(z/h)^3.
# Acima de h: fechamento local de Louis (1979) com funcoes de estabilidade de
# Betts et al. (1996)/NCEP-MRF (eqs. A18-A20 do paper), lambda0=150m
# (Kim & Mahrt 1992, eq. A17).
# SIMPLIFICACOES DOCUMENTADAS: sem contra-gradiente para u,v; sem entranhamento
# de momentum; sem a zona de mistura gaussiana (A13/A14) na transicao h.
# ---------------------------------------------------------------------------
_b_ysu = 7.8       # coeficiente do contragradiente (A3)
_a_ysu = 6.8       # coeficiente do excesso termico (A12)
_lambda0_ysu = 150.0   # comprimento assintotico acima de h (A17)
_K_bg_min_ysu = 0.001*dz   # difusao de fundo minima (0.001*Delta_z, conforme paper)


def _fm_ft_local_ysu(Rig):
    """Funcoes de estabilidade acima de h (Louis 1979 + Betts et al. 1996),
    eqs. A18-A20 do paper original."""
    Rig = np.clip(Rig, -100.0, None)
    ft = np.where(Rig > 0, 1.0/(1.0+5.0*Rig)**2,
                  1.0 - 8.0*Rig/(1.0+1.286*np.sqrt(np.maximum(-Rig, 0.0))))
    fm = np.where(Rig > 0, 1.0/(1.0+5.0*Rig)**2,
                  1.0 - 8.0*Rig/(1.0+1.746*np.sqrt(np.maximum(-Rig, 0.0))))
    return fm, ft


def estagio6_ysu(K_bg=0.3):
    u = np.full(nz, ug); v = np.full(nz, vg); u[0], v[0] = 0.0, 0.0
    theta = theta0_ini + np.where(z <= h0_ini, 0.0, gamma_free*(z-h0_ini))
    Km_prev = np.full(nz, 1.0)

    h_hist, t_hist, theta_hov = [], [], []
    t = 0.0
    h_diag_prev = h0_ini
    for n in range(n_steps):
        Q0 = fluxo_calor_diurno(t)
        dUdz = np.gradient(u, z); dVdz = np.gradient(v, z); dThetadz = np.gradient(theta, z)
        S = np.sqrt(np.maximum(dUdz**2+dVdz**2, 1e-10))

        # --- u* diagnosticado (defasado 1 passo, mesma pratica dos Niveis 4/5)
        u_star2 = np.sqrt((Km_prev[0]*dUdz[0])**2+(Km_prev[0]*dVdz[0])**2)
        u_star = np.sqrt(max(u_star2, 1e-6))

        # --- altura da CLP: usamos o mesmo diagnostico bulk-Ri comum
        #     (simplificacao do procedimento iterativo de duas etapas do
        #     paper original, que usa Ribcr=0 e correcao de theta_T) --------
        h_diag = max(diagnostica_h(u, v, theta, z, theta_va=theta0_ini), 50.0)

        if Q0 > 0:
            # ---------------- CAMADA DE MISTURA (convectiva) --------------
            w_star_b = (g/theta0_ini*Q0*h_diag)**(1.0/3.0)
            idx_h2 = min(int(h_diag/2.0/dz), nz-1)
            ws0 = (u_star**3 + 8.0*k_von*w_star_b**3*0.5)**(1.0/3.0)  # em z=h/2
            ws0 = max(ws0, 1e-3)

            mask = z <= h_diag
            ws = np.where(mask,
                          (u_star**3 + 8.0*k_von*w_star_b**3*np.clip(z/h_diag, 0, 1))**(1.0/3.0),
                          0.0)
            Km = np.where(mask, k_von*ws*z*np.clip(1.0-z/h_diag, 0, None)**2, K_bg)
            Km = np.maximum(Km, K_bg)

            # Prandtl variavel com a altura (eq. A4), Pr0 aproximado (phi~1
            # perto da superficie, simplificacao documentada)
            Pr0 = 0.74 + _b_ysu*k_von*0.1
            Pr = 1.0 + (Pr0-1.0)*np.exp(-3.0*(z-0.1*h_diag)**2/max(h_diag,1.0)**2)
            Pr = np.clip(Pr, 0.25, 4.0)
            Kh = np.maximum(Pr*Km, K_bg)

            gamma_c = _b_ysu*Q0/(ws0*h_diag)   # (A3), so' para theta

            wm3 = w_star_b**3 + 5.0*u_star**3   # (A8/A9), B=5
            wtheta_h = -0.15*(theta0_ini/g)*wm3/h_diag   # (A9)

            # termo assintotico de entranhamento, so' dentro da camada de
            # mistura (z<=h); acima, tratamento local assume o controle
            entrain_z = np.where(mask, wtheta_h*np.clip(z/h_diag, 0, 1)**3, 0.0)

        else:
            # ---------------- ACIMA DE h / REGIME ESTAVEL-NEUTRO -----------
            # (paper: fechamento local em toda a coluna quando (w'theta'_v)_0<0)
            Rig = (g/theta0_ini)*dThetadz/np.maximum(S**2, 1e-10)
            l = 1.0/(1.0/np.maximum(k_von*z, 1e-6) + 1.0/_lambda0_ysu)
            fm, ft = _fm_ft_local_ysu(Rig)
            Km = np.maximum(l**2*fm*S, K_bg)
            Kh = np.maximum(l**2*ft*S, K_bg)
            gamma_c = 0.0
            entrain_z = np.zeros(nz)

        Kh_face = 0.5*(Kh[:-1]+Kh[1:])
        flux_extra_face = Kh_face*gamma_c + 0.5*(entrain_z[:-1]+entrain_z[1:])

        if n % 30 == 0:
            h_hist.append(h_diag); t_hist.append(t/3600.0)
            theta_hov.append(theta.copy())

        u_cor = u + dt*f_cor*(v-vg)
        v_cor = v - dt*f_cor*(u-ug)
        u = difusao_implicita_Kvar(u_cor, Km, dz, dt, phi_surf=0.0)
        v = difusao_implicita_Kvar(v_cor, Km, dz, dt, phi_surf=0.0)
        theta = difusao_implicita_com_fluxo_extra(theta, Kh, dz, dt,
                                                   flux_extra_face=flux_extra_face,
                                                   flux_surf=Q0)
        Km_prev = Km
        t += dt
    return dict(nome="6: YSU completo (nao-local+entranhamento+Louis/Betts)",
                t=np.array(t_hist), h=np.array(h_hist), theta_hov=np.array(theta_hov))
if __name__ == "__main__":
    t0 = timer.time()
    resultados = []
    for fn in [estagio1_ekman, estagio2_obrien, estagio3_my82, estagio4_mynn_diag,
               estagio5_mynn_tke, estagio6_ysu]:
        tA = timer.time()
        r = fn()
        print(f"{r['nome']:45s}  ({timer.time()-tA:5.1f}s)")
        resultados.append(r)
    print(f"\nTempo total: {timer.time()-t0:.1f}s  (N_HORAS={N_HORAS})")

    # -----------------------------------------------------------------
    # FIGURA 1: altura da CLP h(t) para todos os estagios
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    cores = ['gray', 'darkorange', 'steelblue', 'firebrick', 'green', 'purple']
    for r, cor in zip(resultados, cores):
        ax.plot(r['t'], r['h'], color=cor, lw=1.8, label=r['nome'])
    for d in range(int(N_HORAS//24)+1):
        ax.axvspan(24*d+18, 24*d+24+6, color='navy', alpha=0.05)
    ax.set_xlabel('Tempo (h)')
    ax.set_ylabel('Altura da CLP diagnosticada, h (m)')
    ax.set_title(f'Evolucao da CLP ao longo de {N_HORAS:.0f}h - todos os estagios\n'
                 '(faixas azuis = noite, 18h-06h)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/claude/pbl_model/TODOS_h_vs_tempo.png', dpi=140)
    print("Figura 1 (h vs tempo) salva.")

    # -----------------------------------------------------------------
    # FIGURA 2: Hovmoller de theta - Estagio 1 (Ekman) vs Estagio 5 (MYNN-TKE)
    #           (comparacao rapida, so' os dois extremos)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, r, titulo in zip(axes, [resultados[0], resultados[4]],
                              ["Estagio 1: Ekman puro (K=cte)\ndifusao de theta IRREALISTA",
                               "Estagio 5: MYNN + TKE prognostica\ndifusao de theta REALISTA"]):
        im = ax.contourf(r['t'], z, r['theta_hov'].T, levels=30, cmap='RdYlBu_r')
        ax.set_xlabel('Tempo (h)')
        ax.set_title(titulo)
        plt.colorbar(im, ax=ax, label='Theta (K)')
    axes[0].set_ylabel('Altura z (m)')
    for ax in axes:
        ax.set_ylim(0, 1500)
    plt.tight_layout()
    plt.savefig('/home/claude/pbl_model/TODOS_hovmoller_theta.png', dpi=140)
    print("Figura 2 (Hovmoller theta, Ekman vs MYNN-TKE) salva.")

    # -----------------------------------------------------------------
    # FIGURA 2b: Hovmoller de theta - TODOS os 5 estagios lado a lado,
    #            na MESMA escala de cor (para comparacao justa)
    # -----------------------------------------------------------------
    vmin = min(r['theta_hov'].min() for r in resultados)
    vmax = max(r['theta_hov'].max() for r in resultados)
    niveis = np.linspace(vmin, vmax, 31)

    fig, axes = plt.subplots(1, len(resultados), figsize=(4.6*len(resultados), 6), sharey=True)
    for ax, r in zip(axes, resultados):
        im = ax.contourf(r['t'], z, r['theta_hov'].T, levels=niveis, cmap='RdYlBu_r',
                          vmin=vmin, vmax=vmax)
        for d in range(int(N_HORAS//24)+1):
            ax.axvline(24*d, color='k', lw=0.4, alpha=0.3)
        ax.set_xlabel('Tempo (h)')
        ax.set_title(r['nome'], fontsize=10)
        ax.set_ylim(0, 1500)
    axes[0].set_ylabel('Altura z (m)')
    fig.colorbar(im, ax=axes, label='Theta (K)', shrink=0.8, pad=0.01)
    plt.suptitle(f'Hovmoller de theta(z,t) - TODOS OS 5 ESTAGIOS ({N_HORAS:.0f}h, mesma escala de cor)',
                 y=1.03, fontsize=13)
    plt.savefig('/home/claude/pbl_model/TODOS_hovmoller_theta_5estagios.png', dpi=140,
                bbox_inches='tight')
    print("Figura 2b (Hovmoller theta, TODOS os 5 estagios) salva.")

    # -----------------------------------------------------------------
    # RESUMO NUMERICO: h maximo (dia) e h minimo (noite) por estagio
    # -----------------------------------------------------------------
    print("\nResumo (usando o ULTIMO dia completo do ciclo, se N_HORAS>=24):")
    ini_ultimo_dia = max(0, N_HORAS-24)
    for r in resultados:
        mask = r['t'] >= ini_ultimo_dia
        if mask.sum() > 0:
            print(f"  {r['nome']:45s}  h_max={r['h'][mask].max():6.0f} m"
                  f"   h_min={r['h'][mask].min():6.0f} m")
