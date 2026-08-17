import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# =====================================================================================
# EXTENSÃO TEMPORAL (72h) DO MODELO 2D DE GWD (MONTANHA + NUVEM CONVECTIVA)
# -------------------------------------------------------------------------------------
# Fecha o ciclo completo com REALIMENTAÇÃO onda <-> escoamento médio: a cada passo de
# tempo dt,
#     (i)   recalcula a envoltória de amplitude/saturação de Lindzen (Ri_w<1/4,
#           documento consolidado Seção 6.1) de cada fonte usando o U(z) ATUAL
#           -> fluxo de momentum M(z)  (mesma física de source_fields() do modelo
#           2D estático, gwd_model_2d.py — ver ali a dedução completa via a
#           relação de dispersão Eq. 7.43/7.44 e a solução de Queney);
#     (ii)  obtém arrasto ∂u/∂t(z) e aquecimento ∂T/∂t(z) a partir da divergência de
#           M(z)  (termo "gwdd" da equação de Reynolds do momentum, Seção 4);
#     (iii) atualiza U(z) e acumula ΔT(z);
#     (iv)  no PRÓXIMO passo, a fonte e a saturação já usam o U(z) atualizado.
# Ou seja: conforme o jato desacelera, a própria fonte (M0 ∝ U) e o teto de
# saturação (a_crit ∝ U/N) da onda mudam — a onda "desliga" progressivamente a
# si mesma. Para evitar que o jato simplesmente vá a zero e fique estagnado (o que
# aconteceria em uma coluna isolada, sem nenhuma forçante que o regenere), soma-se
# uma relaxação newtoniana ao perfil original U0(z), representando de forma
# simplificada a manutenção do jato pela dinâmica/radiação de grande escala — a
# mesma tensão "arrasto parametrizado vs. dinâmica resolvida" que existe em
# qualquer modelo de PNT/clima operacional.
# =====================================================================================

# ---------- grades ----------
x = np.arange(-150, 150.01, 1.0) * 1000.0
z = np.arange(0, 80.01, 0.25) * 1000.0
nx, nz = len(x), len(z)
dz = z[1] - z[0]

H = 7000.0
rho0 = 1.225
rho = rho0 * np.exp(-z / H)

def smooth_step(z, z0, w):
    """Funcao logistica suave (tanh), soh para construir N(z)/U0(z) idealizados
    sem descontinuidades -- nao corresponde a nenhuma equacao do curso."""
    return 0.5 * (1 + np.tanh((z - z0) / w))

# N(z): N^2 = (g/theta)(d theta/dz) -- Eq. 7.4 (Secao 2.1). Mantido FIXO no tempo
# nesta versao (simplificacao explicita: o acoplamento GWD->estabilidade estatica
# via o proprio aquecimento nao eh fechado aqui).
N = 0.012 + 0.008 * smooth_step(z, 11000, 3000)
# U0(z): perfil de vento de fundo "nao perturbado" -- serve tanto de condicao
# inicial quanto de alvo da relaxacao de grande escala (ver loop temporal).
U0 = (10
      + 20 * smooth_step(z, 3000, 3000) * (1 - smooth_step(z, 11000, 3000))
      + 45 * smooth_step(z, 15000, 8000) * (1 - smooth_step(z, 45000, 8000))
      - 90 * smooth_step(z, 60000, 6000))
cp = 1004.0

# ---------- fontes (mesmos parametros do "Modelo Simples" do curso, Secao 5) ----------
Fc = 1.0
mtn = dict(x0=-70000.0, a=10000.0, h0=250.0, zs=0.0, k=2*np.pi/(2*10000.0))         # Orographic Forcing (a1=10km)
conv = dict(x0=70000.0, a=50000.0, h0=300.0, zs=11000.0, k=2*np.pi/(2*50000.0))     # Convective Forcing (Zt=11km, a2=5a1)
iz_mtn = np.argmin(np.abs(z - mtn["zs"]))
iz_conv = np.argmin(np.abs(z - conv["zs"]))

def column_source(src, iz0, U):
    """Igual a source_fields() de gwd_model_2d.py, mas parametrizada pelo PERFIL
    DE VENTO U(z) fornecido (que muda a cada passo de tempo neste script).
    Retorna: A(z) - envoltoria de amplitude (m, saturacao de Lindzen, Secao 6.1);
             phi(z) - fase vertical acumulada (rad, Φ=∫N/U dz, WKB da Eq. 7.44);
             tau(z) - fluxo de momentum com sinal (~rho*N*U*A^2, Secoes 6.2.1/6.3.1)."""
    A = np.zeros(nz)
    phi = np.zeros(nz)
    Ueff = np.where(np.abs(U) < 3.0, np.sign(U) * 3.0 + (U == 0) * 3.0, U)   # evita divisao por ~0 na fase
    running_cap = np.inf
    rho_src = rho[iz0]
    for i in range(iz0, nz):
        env = src["h0"] * np.sqrt(rho_src / rho[i])                # crescimento de amplitude por densidade
        a_crit = Fc * abs(U[i]) / max(N[i], 1e-6)                  # amplitude critica de Lindzen (Ri_w<1/4, Secao 6.1)
        running_cap = min(running_cap, a_crit)
        A[i] = min(env, running_cap)                               # saturacao irreversivel
        if i > iz0:
            phi[i] = phi[i-1] + N[i] / Ueff[i] * dz                 # Φ(z)=∫N/U dz (fase vertical, WKB da Eq. 7.44)
    M = 0.5 * rho * N * np.abs(U) * A**2 * src["k"]                 # fluxo de momentum (~rho*N*U*A^2)
    sign0 = np.sign(U[iz0]) if U[iz0] != 0 else 1.0
    tau = np.where(np.arange(nz) >= iz0, -sign0 * M, 0.0)           # estresse com sinal, oposto ao vento na fonte
    return A, phi, tau

def smooth(v, win=25):
    """Suavizacao com padding por borda -- representa uma camada fisica de
    quebra (nao ruido de grade)."""
    kernel = np.ones(win) / win
    pad = win // 2
    return np.convolve(np.pad(v, pad, mode="edge"), kernel, mode="valid")

def drag_from_tau(tau):
    """du/dt = -(1/rho) d(tau)/dz -- termo 'gwdd' da equacao de Reynolds do
    momentum (documento consolidado, Secao 4)."""
    d = np.zeros(nz)
    d[1:-1] = -(1.0 / rho[1:-1]) * (tau[2:] - tau[:-2]) / (2 * dz)
    d[0], d[-1] = d[1], d[-2]
    return np.clip(d, -6e-4, 6e-4)          # limitador numerico (~52 m/s/dia), soh p/ estabilidade

def tendencies(U):
    """Combina as duas fontes (superposicao linear) e retorna:
    du/dt(z), dT/dt(z) = -(U/cp)*du/dt (extensao ilustrativa, Secao 4),
    mais os pares (A,phi) de cada fonte, reaproveitados depois para desenhar
    o campo 2D eta(x,z) nos instantaneos (eta_field)."""
    A_m, phi_m, tau_m = column_source(mtn, iz_mtn, U)
    A_c, phi_c, tau_c = column_source(conv, iz_conv, U)
    tau_m_s, tau_c_s = smooth(tau_m), smooth(tau_c)
    dudt = drag_from_tau(tau_m_s) + drag_from_tau(tau_c_s)
    dTdt = -(U / cp) * dudt
    return dudt, dTdt, (A_m, phi_m), (A_c, phi_c)

def eta_field(src, iz0, A, phi):
    """Campo 2D de deslocamento de isentropica/streamline -- solucao de Queney
    (1948) para a montanha de Agnesi, com amplitude A(z) e fase Φ(z) ja
    calculadas por column_source() (mesma formula de gwd_model_2d.py):
        eta(x,z) = A(z)*a*[a*cos(Φ) - (x-x0)*sin(Φ)] / (a² + (x-x0)²)
    """
    a = src["a"]
    dx = x - src["x0"]
    eta = np.zeros((nz, nx))
    for i in range(iz0, nz):
        eta[i, :] = A[i] * a * (a * np.cos(phi[i]) - dx * np.sin(phi[i])) / (a**2 + dx**2)
    return eta

# ---------- integracao no tempo: 72h ----------
dt = 900.0                       # 15 min
hours_total = 72
nsteps = int(hours_total * 3600 / dt)
save_every = int(3600 / dt)      # guarda 1 vez por hora

U = U0.copy()
tau_relax = 4.0 * 86400.0        # tempo de relaxacao ao jato de fundo (proxy p/ forcante de grande escala/radiativa, ~4 dias)
t_hours = [0.0]
U_hist = [U.copy()]
dT_accum = np.zeros(nz)
dT_hist = [dT_accum.copy()]

snapshot_hours = [0, 24, 48, 72]
snapshots = {}

for step in range(1, nsteps + 1):
    dudt, dTdt, srcm, srcc = tendencies(U)                 # (i)+(ii): GWD calculado com o U(z) ATUAL
    dudt_relax = (U0 - U) / tau_relax                      # restauracao newtoniana de grande escala
    dU = (dudt + dudt_relax) * dt                          # (iii): passo de Euler explicito
    dU = np.clip(dU, -0.08 * np.abs(U) - 0.02, 0.08 * np.abs(U) + 0.02)   # limitador (~espirito do CB98/CB02, Secao 6.3.2)
    U = U + dU                                             # (iv): U(z) atualizado -> proximo passo usa fonte/saturacao novas
    dT_accum = dT_accum + dTdt * dt                        # acumulo de temperatura (integral simples no tempo)

    t_now = step * dt / 3600.0
    if step % save_every == 0:
        t_hours.append(t_now)
        U_hist.append(U.copy())
        dT_hist.append(dT_accum.copy())

    for sh in snapshot_hours:
        if abs(t_now - sh) < 1e-6 and sh not in snapshots:
            _, _, srcm2, srcc2 = tendencies(U)
            eta = eta_field(mtn, iz_mtn, *srcm2) + eta_field(conv, iz_conv, *srcc2)
            snapshots[sh] = dict(U=U.copy(), eta=eta, dudt=dudt.copy())

if 0 not in snapshots:
    dudt0, _, srcm0, srcc0 = tendencies(U0)
    eta0 = eta_field(mtn, iz_mtn, *srcm0) + eta_field(conv, iz_conv, *srcc0)
    snapshots[0] = dict(U=U0.copy(), eta=eta0, dudt=dudt0.copy())

U_hist = np.array(U_hist)          # (ntime, nz)
dT_hist = np.array(dT_hist)
t_hours = np.array(t_hours)
z_km, x_km = z / 1000.0, x / 1000.0

# =====================================================================
# FIGURA 1: Hovmoller (tempo x altura) do vento e da temperatura acumulada
# =====================================================================
fig, axs = plt.subplots(1, 4, figsize=(18, 7), sharey=True)

cf0 = axs[0].contourf(t_hours, z_km, U_hist.T, levels=25, cmap="RdBu_r", vmin=-90, vmax=90)
axs[0].set_title("(a) U(z,t) — vento zonal")
axs[0].set_xlabel("tempo (h)"); axs[0].set_ylabel("Altura (km)")
fig.colorbar(cf0, ax=axs[0], shrink=0.85, label="m/s")

cf1 = axs[1].contourf(t_hours, z_km, dT_hist.T, levels=25, cmap="YlOrRd")
axs[1].set_title("(b) ΔT(z,t) acumulado")
axs[1].set_xlabel("tempo (h)")
fig.colorbar(cf1, ax=axs[1], shrink=0.85, label="K")

for zt, c in zip([30, 40, 50], ["#1F3864", "#B22222", "#2E7D32"]):
    i = np.argmin(np.abs(z_km - zt))
    axs[2].plot(t_hours, U_hist[:, i], color=c, lw=2, label=f"{zt} km")
axs[2].set_title("(c) U no tempo\n(níveis fixos)")
axs[2].set_xlabel("tempo (h)"); axs[2].legend(fontsize=8)
axs[2].axhline(0, color="gray", lw=0.6)
axs[2].grid(alpha=0.3)

for zt, c in zip([30, 40, 50], ["#1F3864", "#B22222", "#2E7D32"]):
    i = np.argmin(np.abs(z_km - zt))
    axs[3].plot(t_hours, dT_hist[:, i], color=c, lw=2, label=f"{zt} km")
axs[3].set_title("(d) ΔT acumulado\n(níveis fixos)")
axs[3].set_xlabel("tempo (h)"); axs[3].legend(fontsize=8)
axs[3].grid(alpha=0.3)

for ax in axs:
    ax.axhspan(50, 80, color="orange", alpha=0.06)
    ax.axhspan(11, 50, color="steelblue", alpha=0.04)
    ax.set_ylim(0, 80)

fig.suptitle("Evolução temporal (72h) do modelo de GWD: montanha + nuvem convectiva", fontsize=13)
plt.tight_layout()
plt.savefig("/home/claude/doc/gwd_2d_time_hovmoller.png", dpi=140, bbox_inches="tight")
print("saved hovmoller")

# =====================================================================
# FIGURA 2: instantaneos do campo de onda em t=0,24,48,72h
# =====================================================================
fig2, axs2 = plt.subplots(1, 4, figsize=(20, 7), sharey=True)
lvl = np.linspace(-350, 350, 29)
for ax, sh in zip(axs2, snapshot_hours):
    s = snapshots[sh]
    cf = ax.contourf(x_km, z_km, s["eta"], levels=lvl, cmap="RdBu_r", extend="both")
    h_mtn_profile = mtn["h0"]/1000 * 8 * (mtn["a"]**2 / (mtn["a"]**2 + (x - mtn["x0"])**2))
    ax.fill_between(x_km, 0, h_mtn_profile, color="#4b3621", zorder=5)
    for dx_, dz_, r in [(-8,0,14),(0,4,18),(9,0,15),(-3,9,12),(5,9,11)]:
        ax.add_patch(Ellipse((70+dx_, 6+dz_), r, r*0.9, color="#666666", alpha=0.5, zorder=5))
    ax.set_title(f"t = {sh} h")
    ax.set_xlabel("x (km)")
    ax.set_ylim(0, 80); ax.set_xlim(-150, 150)
    ax.axhspan(50, 80, color="orange", alpha=0.06)
axs2[0].set_ylabel("Altura (km)")
fig2.suptitle("Campo de onda η(x,z) em 4 instantes — jato enfraquece, fonte e alcance vertical da onda diminuem",
              fontsize=12.5)
fig2.colorbar(cf, ax=axs2, shrink=0.7, label="deslocamento de isentrópica (m)")
plt.savefig("/home/claude/doc/gwd_2d_snapshots.png", dpi=140, bbox_inches="tight")
print("saved snapshots")

for zt in [30, 40, 50]:
    i = np.argmin(np.abs(z_km - zt))
    print(f"{zt} km: U(0h)={U_hist[0,i]:.1f}  U(24h)={U_hist[24,i]:.1f}  U(48h)={U_hist[48,i]:.1f}  U(72h)={U_hist[-1,i]:.1f} m/s | "
          f"ΔT(72h)={dT_hist[-1,i]:.2f} K")
