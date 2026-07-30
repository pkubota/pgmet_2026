"""
Modelo Termodinâmico de Solo — Fluxo de Calor [G]
===================================================
Equações do slide (Paulo Yoshio Kubota):

  C(θ_k) ∂T/∂t = ∂/∂z [ K(θ) ∂T/∂z ]

Discretizado pelo esquema totalmente implícito:

  C(θ̃_k) (T̃_k^{n+1} - T̃_k^n)/Δt =
      1/(Δz)_k [ K(θ_{k-1})(∂T/∂z)^{n+1}|z_{k-1}
               - K(θ_k  )(∂T/∂z)^{n+1}|z_k       ]

Condições de contorno:
  Topo  : temperatura superficial T_s(t)  [BC Dirichlet – ciclo diurno]
  Fundo : temperatura constante   T_bot   [BC Dirichlet]

Referência de gradiente no topo (k=0):
  ∂T/∂z|_{z0} = (T̃_s - T̃_1) / (0.5 * Δz̃_1)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parâmetros do modelo ─────────────────────────────────────────────────────

NZ      = 15          # número de camadas de solo
DZ      = 0.10        # espessura de cada camada [m]
DT      = 1800        # passo de tempo [s]   (30 min)
NDAYS   = 3           # dias de simulação

T_INIT  = 22.0        # temperatura inicial uniforme [°C]
T_BOT   = 15.0        # temperatura no fundo (constante) [°C]
T_MEAN  = 22.0        # média da temperatura superficial [°C]
T_AMP   = 12.0        # amplitude do ciclo diurno [°C]

K_SOLO  = 0.50        # condutividade térmica [W m⁻¹ K⁻¹]
C_SOLO  = 1.4e6       # capacidade calorífica volumétrica [J m⁻³ K⁻¹]

# ── Malha ────────────────────────────────────────────────────────────────────

dz  = np.full(NZ, DZ)                       # espessura de cada camada [m]
z_c = np.cumsum(dz) - dz / 2                # profundidade dos centros [m]
z_i = np.concatenate([[0.0], np.cumsum(dz)])# profundidade das interfaces [m]

# Propriedades (podem ser arrays heterogêneos)
K = np.full(NZ, K_SOLO)   # condutividade por camada [W m⁻¹ K⁻¹]
C = np.full(NZ, C_SOLO)   # capacidade calorífica   [J m⁻³ K⁻¹]

# Condutividade nas interfaces (média harmônica entre camadas adjacentes)
K_int = np.zeros(NZ + 1)
for i in range(1, NZ):
    K_int[i] = 2 * K[i-1] * K[i] / (K[i-1] + K[i])
K_int[0]  = K[0]     # interface topo  → usa camada superior
K_int[-1] = K[-1]    # interface fundo → usa camada inferior

# ── Algoritmo de Thomas (solução de sistema tridiagonal) ─────────────────────

def thomas(a, b, c, d):
    """
    Resolve  A x = d  onde A é tridiagonal.
    a : subdiagonal  (a[0] ignorado)
    b : diagonal principal
    c : superdiagonal (c[-1] ignorado)
    d : lado direito
    """
    n  = len(b)
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    x  = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        m    = b[i] - a[i] * c_[i-1]
        c_[i] = c[i] / m
        d_[i] = (d[i] - a[i] * d_[i-1]) / m

    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]
    return x

# ── Montagem do sistema implícito ────────────────────────────────────────────

def montar_sistema(T, Ts):
    """
    Monta o sistema tridiagonal para o passo implícito.
    Retorna vetores a (sub), b (diag), c (super), rhs.
    """
    a   = np.zeros(NZ)
    b   = np.zeros(NZ)
    c   = np.zeros(NZ)
    rhs = np.zeros(NZ)

    for k in range(NZ):
        dz_k = dz[k]
        C_k  = C[k]

        # ── interface superior da camada k (z_{k-1})
        if k == 0:
            dz_up = 0.5 * dz[k]          # distância centro → topo (BC)
        else:
            dz_up = 0.5 * (dz[k-1] + dz[k])

        # ── interface inferior da camada k (z_k)
        if k == NZ - 1:
            dz_dn = 0.5 * dz[k]          # distância centro → fundo (BC)
        else:
            dz_dn = 0.5 * (dz[k] + dz[k+1])

        alpha_up = K_int[k]   / (dz_k * dz_up)
        alpha_dn = K_int[k+1] / (dz_k * dz_dn)

        b[k]  = C_k / DT + alpha_up + alpha_dn
        rhs[k] = C_k / DT * T[k]

        if k > 0:
            a[k] = -alpha_up          # coef. de T_{k-1}
        else:
            rhs[k] += alpha_up * Ts   # BC topo → Dirichlet

        if k < NZ - 1:
            c[k] = -alpha_dn          # coef. de T_{k+1}
        else:
            rhs[k] += alpha_dn * T_BOT  # BC fundo → Dirichlet

    return a, b, c, rhs

# ── Temperatura superficial (ciclo diurno) ───────────────────────────────────

def T_superficie(t_s):
    """Ciclo diurno senoidal; máximo às 14h, mínimo às 02h."""
    return T_MEAN + T_AMP * np.sin(2 * np.pi * t_s / 86400 - np.pi / 2)

# ── Integração temporal ───────────────────────────────────────────────────────

NT = int(NDAYS * 86400 / DT)
T  = np.full(NZ, T_INIT)

T_hist  = np.zeros((NT, NZ))    # temperatura [°C]
Ts_hist = np.zeros(NT)          # temperatura superficial [°C]
G_hist  = np.zeros(NT)          # fluxo de calor no solo [W m⁻²]
time_h  = np.zeros(NT)          # tempo [h]

for n in range(NT):
    t_s       = n * DT
    time_h[n] = t_s / 3600
    Ts        = T_superficie(t_s)
    Ts_hist[n] = Ts

    # Fluxo G = –K ∂T/∂z|_{z=0}  (positivo = para baixo)
    dTdz_sup  = (T[0] - Ts) / (0.5 * dz[0])
    G_hist[n] = -K[0] * dTdz_sup

    a, b, c, rhs = montar_sistema(T, Ts)
    T = thomas(a, b, c, rhs)
    T_hist[n] = T.copy()

# ── Visualização ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Modelo Termodinâmico de Solo — Fluxo de Calor [G]",
             fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ── (1) Temperatura superficial ──────────────────────────────────────────────
ax1.plot(time_h, Ts_hist, color="#c0392b", lw=2)
ax1.set_xlabel("Tempo (h)")
ax1.set_ylabel("T_s (°C)")
ax1.set_title("Temperatura Superficial")
ax1.grid(True, ls="--", alpha=0.5)
ax1.set_xlim(0, NT * DT / 3600)

# ── (2) Série temporal de temperatura em diferentes profundidades ─────────────
cmap  = plt.get_cmap("plasma")
layer_sel = [0, 2, 5, 9, 14]
for idx, k in enumerate(layer_sel):
    cor = cmap(idx / (len(layer_sel) - 1))
    ax2.plot(time_h, T_hist[:, k], color=cor, lw=1.8,
             label=f"z = {z_c[k]:.2f} m")
ax2.set_xlabel("Tempo (h)")
ax2.set_ylabel("Temperatura (°C)")
ax2.set_title("Temperatura por Profundidade")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, ls="--", alpha=0.5)
ax2.set_xlim(0, NT * DT / 3600)

# ── (3) Perfil vertical em diferentes horários (último dia) ──────────────────
nt_per_day = int(86400 / DT)
t_start    = (NDAYS - 1) * nt_per_day   # início do último dia
n_snaps    = 8
snap_idx   = np.linspace(t_start, t_start + nt_per_day - 1, n_snaps, dtype=int)
cmap2 = plt.get_cmap("coolwarm")
for j, ni in enumerate(snap_idx):
    hora = time_h[ni] % 24
    cor  = cmap2(j / (n_snaps - 1))
    ax3.plot(T_hist[ni], -z_c, color=cor, lw=1.8, label=f"{hora:04.1f}h")
ax3.set_xlabel("Temperatura (°C)")
ax3.set_ylabel("Profundidade (m)")
ax3.set_title(f"Perfil Vertical — Dia {NDAYS}")
ax3.legend(fontsize=7, loc="lower right", ncol=2)
ax3.grid(True, ls="--", alpha=0.5)

# ── (4) Fluxo de calor no solo G ─────────────────────────────────────────────
ax4.plot(time_h, G_hist, color="#2980b9", lw=1.8)
ax4.axhline(0, color="k", lw=0.8, ls="--")
ax4.fill_between(time_h, G_hist, 0,
                 where=(G_hist > 0), alpha=0.25, color="#e74c3c",
                 label="G > 0 (para baixo)")
ax4.fill_between(time_h, G_hist, 0,
                 where=(G_hist < 0), alpha=0.25, color="#3498db",
                 label="G < 0 (para cima)")
ax4.set_xlabel("Tempo (h)")
ax4.set_ylabel("G (W m⁻²)")
ax4.set_title("Fluxo de Calor no Solo (G)")
ax4.legend(fontsize=8)
ax4.grid(True, ls="--", alpha=0.5)
ax4.set_xlim(0, NT * DT / 3600)

plt.savefig("/mnt/user-data/outputs/solo_calor.png", dpi=150, bbox_inches="tight")
print("Figura salva: solo_calor.png")

# ── Estatísticas básicas ──────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"  NZ={NZ} camadas  |  Δz={DZ} m  |  Δt={DT} s  |  {NDAYS} dias")
print(f"  K = {K_SOLO} W/m/K    C = {C_SOLO:.1e} J/m³/K")
print(f"  T_bot = {T_BOT} °C      T_sup ∈ [{T_MEAN-T_AMP:.1f}, {T_MEAN+T_AMP:.1f}] °C")
print(f"  G_max = {G_hist.max():.1f} W/m²   G_min = {G_hist.min():.1f} W/m²")
print(f"{'─'*50}")
