"""
Laplace vs Poisson — Contexto Meteorológico
============================================
Equação de Laplace : nabla²psi = 0        → escoamento geostrófico puro
                                              (sem vorticidade relativa)
Equação de Poisson : nabla²psi = -zeta(x,y) → função de corrente com
                                              ciclone extratropical

Física:
  psi  = função de corrente geostrófica  [m²/s adimensional aqui: psi/(U0*L)]
  zeta = vorticidade relativa             [s⁻¹]
  Vento geostrófico:
      u_geo = -dpsi/dy
      v_geo = +dpsi/dx

Domínio : [0, L] × [0, L] com L = 6000 km (escala sinótica)
Solver   : diferenças finitas centradas, sistema linear direto

Dependências : numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch

# ── Parâmetros físicos ────────────────────────────────────────────────────────
L      = 6.0e6      # extensão do domínio [m]  (6000 km)
U0     = 15.0       # vento zonal de fundo [m/s]
f0     = 1.0e-4     # parâmetro de Coriolis a ~45°N [s⁻¹]

# Ondulação do jato (condição de contorno norte)
A_BC   = 0.25       # amplitude da onda (fração de psi_N)

# Ciclone extratropical — apenas no caso Poisson
X_CYC  = 0.42       # posição x (adimensional, 0–1)
Y_CYC  = 0.52       # posição y (adimensional, 0–1)
SIGMA  = 0.12       # raio do ciclone (adimensional)
ZETA0  = 5.0        # vorticidade máxima adimensional = zeta * L / U0
                    # (5 → zeta_real ~ 5*15/6e6 = 1.25e-5 s⁻¹, valor sinótico típico)

# Anticiclone — apenas no caso Poisson (adicionado ao NE do domínio)
X_ANT  = 0.72
Y_ANT  = 0.60
ZETA_ANT = -3.0     # negativo = anticiclônico

# Resolução da malha interna
N      = 70

# ── Paleta ────────────────────────────────────────────────────────────────────
BG    = '#fdf6e3'
BLUE  = '#1a3a6e'
RED   = '#c0392b'
TEAL  = '#0d5e52'
DGRAY = '#444444'

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def bc_psi(xi, yi):
    """
    Condições de contorno para psi adimensional.
      Sul  (y=0): psi = 0
      Norte(y=1): psi = 1 + A_BC*sin(2*pi*x)   ← jato ondulado
      Oeste(x=0): psi = y
      Leste(x=1): psi = y
    """
    if   yi <= 1e-12:            return 0.0
    elif yi >= 1.0 - 1e-12:      return 1.0 + A_BC * np.sin(2 * np.pi * xi)
    else:                        return float(yi)   # oeste e leste


def vorticity_source(xi, yi, include_cyclone):
    """
    Termo fonte adimensional: zeta_tilde = zeta * L / U0
    Gaussiana positiva (ciclone) + gaussiana negativa (anticiclone).
    """
    if not include_cyclone:
        return 0.0
    r2_cyc = (xi - X_CYC)**2 + (yi - Y_CYC)**2
    r2_ant = (xi - X_ANT)**2 + (yi - Y_ANT)**2
    return (ZETA0  * np.exp(-r2_cyc / (2 * SIGMA**2))
          + ZETA_ANT * np.exp(-r2_ant / (2 * SIGMA**2)))


def solve_streamfunction(N=60, include_cyclone=False):
    """
    Resolve  nabla²psi = -zeta(x,y)  em [0,1]² via diferenças finitas.

    O operador nabla² em coordenadas normalizadas corresponde a:
        (1/h²) * [psi(i+1,j) + psi(i-1,j) + psi(i,j+1) + psi(i,j-1) - 4*psi(i,j)]
                = -zeta_tilde(xi, yi)

    Portanto o RHS no sistema linear é: h² * (-zeta_tilde).

    Retorna
    -------
    X, Y : arrays (N+2)×(N+2)  coordenadas adimensionais
    PSI  : array  (N+2)×(N+2)  função de corrente adimensional
    ZETA_SRC : array campo de vorticidade fonte (para visualização)
    """
    h = 1.0 / (N + 1)
    x = np.linspace(0, 1, N + 2)
    y = np.linspace(0, 1, N + 2)
    X, Y = np.meshgrid(x, y)

    size  = N * N
    A_mat = np.zeros((size, size))
    b_vec = np.zeros(size)

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            xi = x[j + 1]
            yi = y[i + 1]
            k  = idx(i, j)

            A_mat[k, k] = -4.0

            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    A_mat[k, idx(ni, nj)] = 1.0
                else:
                    bxi = x[nj + 1] if 0 <= nj < N else (0.0 if nj < 0 else 1.0)
                    byi = y[ni + 1] if 0 <= ni < N else (0.0 if ni < 0 else 1.0)
                    b_vec[k] -= bc_psi(bxi, byi)

            # RHS: -zeta_tilde * h²
            zeta = vorticity_source(xi, yi, include_cyclone)
            b_vec[k] += h**2 * (-zeta)

    psi_inner = np.linalg.solve(A_mat, b_vec)

    PSI = np.zeros((N + 2, N + 2))
    for i in range(N):
        for j in range(N):
            PSI[i + 1, j + 1] = psi_inner[idx(i, j)]

    # Aplicar BCs nas bordas
    PSI[0,  :] = 0.0
    PSI[-1, :] = 1.0 + A_BC * np.sin(2 * np.pi * x)
    PSI[:,  0] = y
    PSI[:, -1] = y

    # Campo de vorticidade fonte (para plotar)
    ZETA_SRC = np.zeros_like(PSI)
    for i in range(N + 2):
        for j in range(N + 2):
            ZETA_SRC[i, j] = vorticity_source(x[j], y[i], include_cyclone)

    return X, Y, PSI, ZETA_SRC


def compute_winds(PSI, h):
    """
    Vento geostrófico adimensional a partir de psi:
        u_nd = -dpsi/dy   →  u_phys = U0 * u_nd
        v_nd = +dpsi/dx   →  v_phys = U0 * v_nd
    Usa diferenças centradas (np.gradient usa espaçamento h).
    """
    dpsi_dy, dpsi_dx = np.gradient(PSI, h)
    u_nd = -dpsi_dy   # u = -dpsi/dy
    v_nd =  dpsi_dx   # v = +dpsi/dx
    return u_nd * U0, v_nd * U0   # [m/s]


def compute_vorticity_fd(PSI, h):
    """
    Calcula a vorticidade zeta = nabla²psi via Laplaciano de diferenças finitas.
    Retorna em unidades físicas: [s⁻¹]  (mult. por U0/L).
    """
    N2 = PSI.shape[0]
    ZETA = np.zeros_like(PSI)
    for i in range(1, N2 - 1):
        for j in range(1, N2 - 1):
            ZETA[i, j] = (PSI[i+1,j] + PSI[i-1,j]
                        + PSI[i,j+1] + PSI[i,j-1]
                        - 4*PSI[i,j]) / h**2
    return ZETA * (U0 / L)   # [s⁻¹]


# ─────────────────────────────────────────────────────────────────────────────
# RESOLVER
# ─────────────────────────────────────────────────────────────────────────────
print("Resolvendo Laplace (sem vorticidade)...")
X, Y, PSI_lap, _ = solve_streamfunction(N, include_cyclone=False)

print("Resolvendo Poisson (ciclone + anticiclone)...")
X, Y, PSI_poi, ZETA_SRC = solve_streamfunction(N, include_cyclone=True)

h_nd = 1.0 / (N + 1)

U_lap, V_lap = compute_winds(PSI_lap, h_nd)
U_poi, V_poi = compute_winds(PSI_poi, h_nd)

ZETA_lap = compute_vorticity_fd(PSI_lap, h_nd)
ZETA_poi = compute_vorticity_fd(PSI_poi, h_nd)

# Coordenadas em km para os eixos
Xkm = X * (L / 1e3)
Ykm = Y * (L / 1e3)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURA
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 20), facecolor=BG)
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(4, 2, figure=fig,
                        hspace=0.52, wspace=0.35,
                        top=0.95, bottom=0.04,
                        left=0.07, right=0.97)

# ── Painel 0 — equações e contexto físico ────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.set_facecolor(BG); ax0.axis('off')

# caixa Laplace
ax0.add_patch(Rectangle((0.01, 0.05), 0.455, 0.90,
    transform=ax0.transAxes, facecolor='#e8f0fb', edgecolor=BLUE, lw=2))
ax0.text(0.235, 0.87, 'Equacao de Laplace',
         transform=ax0.transAxes, ha='center', fontsize=14,
         color=BLUE, fontweight='bold', fontfamily='serif')
ax0.text(0.235, 0.62, r'$\nabla^2\psi = 0$',
         transform=ax0.transAxes, ha='center', fontsize=22, color=BLUE)
ax0.text(0.235, 0.34,
         'Escoamento geostrof. puro\n'
         r'$u_g = -\partial\psi/\partial y$'
         '    '
         r'$v_g = +\partial\psi/\partial x$',
         transform=ax0.transAxes, ha='center', fontsize=10.5, color=BLUE)
ax0.text(0.235, 0.12,
         'Sem ciclones/anticiclones → linhas de corrente\n'
         'suaves, sem extremos no interior',
         transform=ax0.transAxes, ha='center', fontsize=9.5,
         color='#4a6090', style='italic')

# caixa Poisson
ax0.add_patch(Rectangle((0.535, 0.05), 0.455, 0.90,
    transform=ax0.transAxes, facecolor='#fdecea', edgecolor=RED, lw=2))
ax0.text(0.765, 0.87, 'Equacao de Poisson',
         transform=ax0.transAxes, ha='center', fontsize=14,
         color=RED, fontweight='bold', fontfamily='serif')
ax0.text(0.765, 0.62, r'$\nabla^2\psi = -\zeta(x,y)$',
         transform=ax0.transAxes, ha='center', fontsize=22, color=RED)
ax0.text(0.765, 0.34,
         r'$\zeta > 0$: ciclone (B. Pressao)    '
         r'$\zeta < 0$: anticiclone (A. Pressao)',
         transform=ax0.transAxes, ha='center', fontsize=10.5, color=RED)
ax0.text(0.765, 0.12,
         'A vorticidade forcada deforma as linhas de corrente\n'
         'criando centros de baixa e alta pressao no interior',
         transform=ax0.transAxes, ha='center', fontsize=9.5,
         color='#903030', style='italic')

# ── Paineis 1–2 — função de corrente + ventos ────────────────────────────────
psi_levels = np.linspace(0, 1.3, 24)
skip = 5   # decimação do quiver

def add_psi_panel(ax, Xkm, Ykm, PSI, U, V, title, color, note=''):
    cf = ax.contourf(Xkm, Ykm, PSI, levels=psi_levels,
                     cmap='RdYlBu_r', alpha=0.85)
    cs = ax.contour(Xkm, Ykm, PSI, levels=psi_levels,
                    colors='white', linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fmt='%.2f', fontsize=7, colors='white', inline=True)
    plt.colorbar(cf, ax=ax, shrink=0.88, label=r'$\psi$ (adim.)')

    Xs = Xkm[::skip, ::skip]
    Ys = Ykm[::skip, ::skip]
    Us = U  [::skip, ::skip]
    Vs = V  [::skip, ::skip]
    speed = np.sqrt(Us**2 + Vs**2)
    ax.quiver(Xs, Ys, Us/speed, Vs/speed, speed,
              cmap='Greys_r', scale=30, width=0.004,
              alpha=0.75, clim=[0, 30])

    ax.set_title(title, fontsize=11, color=color, fontweight='bold', pad=6)
    ax.set_xlabel('x  [km]'); ax.set_ylabel('y  [km]')
    ax.set_xlim(0, L/1e3); ax.set_ylim(0, L/1e3)
    if note:
        ax.text(0.5, -0.10, note, transform=ax.transAxes,
                ha='center', fontsize=8.5, color=DGRAY, style='italic')

ax1 = fig.add_subplot(gs[1, 0]); ax1.set_facecolor(BG)
add_psi_panel(ax1, Xkm, Ykm, PSI_lap, U_lap, V_lap,
              r'$\nabla^2\psi = 0$ — Escoamento geostrofico puro', BLUE,
              'Jato ondulado na fronteira norte; sem perturbacoes internas')

ax2 = fig.add_subplot(gs[1, 1]); ax2.set_facecolor(BG)
add_psi_panel(ax2, Xkm, Ykm, PSI_poi, U_poi, V_poi,
              r'$\nabla^2\psi = -\zeta$ — Ciclone + Anticiclone', RED,
              r'Ciclone ($\zeta>0$) a SW e anticiclone ($\zeta<0$) a NE')

# marcadores C e A no painel Poisson
ax2.text(X_CYC*L/1e3, Y_CYC*L/1e3, 'B', fontsize=16, fontweight='bold',
         color='white', ha='center', va='center',
         bbox=dict(facecolor=RED, boxstyle='circle,pad=0.3', alpha=0.85))
ax2.text(X_ANT*L/1e3, Y_ANT*L/1e3, 'A', fontsize=16, fontweight='bold',
         color='white', ha='center', va='center',
         bbox=dict(facecolor=BLUE, boxstyle='circle,pad=0.3', alpha=0.85))

# ── Paineis 3–4 — vorticidade relativa ───────────────────────────────────────
vmax = max(np.abs(ZETA_poi).max(), 1e-6)
norm_z = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
zeta_levels = np.linspace(-vmax, vmax, 24)

def add_zeta_panel(ax, Xkm, Ykm, ZETA, title, color, note=''):
    cf = ax.contourf(Xkm, Ykm, ZETA * 1e5, levels=zeta_levels*1e5,
                     cmap='bwr', alpha=0.88)
    ax.contour(Xkm, Ykm, ZETA * 1e5, levels=[0],
               colors='black', linewidths=1.2, linestyles='--')
    cb = plt.colorbar(cf, ax=ax, shrink=0.88)
    cb.set_label(r'$\zeta \times 10^{-5}$  [s$^{-1}$]')
    ax.set_title(title, fontsize=11, color=color, fontweight='bold', pad=6)
    ax.set_xlabel('x  [km]'); ax.set_ylabel('y  [km]')
    ax.set_xlim(0, L/1e3); ax.set_ylim(0, L/1e3)
    if note:
        ax.text(0.5, -0.10, note, transform=ax.transAxes,
                ha='center', fontsize=8.5, color=DGRAY, style='italic')

ax3 = fig.add_subplot(gs[2, 0]); ax3.set_facecolor(BG)
add_zeta_panel(ax3, Xkm, Ykm, ZETA_lap,
               r'Vorticidade $\zeta = \nabla^2\psi \approx 0$  [Laplace]',
               BLUE, 'Campo quase nulo — confirma o principio do maximo')

ax4 = fig.add_subplot(gs[2, 1]); ax4.set_facecolor(BG)
add_zeta_panel(ax4, Xkm, Ykm, ZETA_poi,
               r'Vorticidade $\zeta = \nabla^2\psi$  [Poisson]',
               RED,
               r'Ciclone (vermelho, $\zeta>0$)  e  anticiclone (azul, $\zeta<0$)')

# ── Painel 5 — perfil zonal de u no meio do domínio ──────────────────────────
ax5 = fig.add_subplot(gs[3, :])
ax5.set_facecolor('#f5f7fd')

jmid = N // 2   # linha y ≈ 0.5
x_km = Xkm[0, :]

ax5.plot(x_km, U_lap[jmid, :], color=BLUE, lw=2.5, label=r'$u_g$  (Laplace, $\nabla^2\psi=0$)')
ax5.plot(x_km, U_poi[jmid, :], color=RED,  lw=2.5, label=r'$u_g$  (Poisson, com ciclone)')
ax5.axhline(U0, color='gray', lw=1.2, ls='--', label=f'Vento de fundo $U_0 = {U0:.0f}$ m/s')
ax5.axhline(0,  color='black', lw=0.8, ls=':')

ax5.fill_between(x_km, U_lap[jmid, :], U_poi[jmid, :],
                 alpha=0.15, color='purple',
                 label='Perturbacao pelo ciclone')

ax5.set_xlabel('x  [km]', fontsize=11)
ax5.set_ylabel(r'$u_g$  [m/s]', fontsize=11)
ax5.set_title(r'Perfil zonal do vento geostrof. $u_g(x)$ em $y = L/2$ (meio do dominio)',
              fontsize=11, color=DGRAY, fontweight='bold')
ax5.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
ax5.set_xlim(0, L/1e3)
ax5.grid(True, alpha=0.3)

# anotação ciclone
ax5.axvline(X_CYC*L/1e3, color=RED, lw=1.0, ls='--', alpha=0.6)
ax5.text(X_CYC*L/1e3, ax5.get_ylim()[0]*0.95 if ax5.get_ylim()[0]<0 else 2,
         'Ciclone', color=RED, fontsize=8.5, ha='center', style='italic')
ax5.axvline(X_ANT*L/1e3, color=BLUE, lw=1.0, ls='--', alpha=0.6)
ax5.text(X_ANT*L/1e3, ax5.get_ylim()[0]*0.95 if ax5.get_ylim()[0]<0 else 2,
         'Anticicl.', color=BLUE, fontsize=8.5, ha='center', style='italic')

# ── Título geral ─────────────────────────────────────────────────────────────
plt.suptitle(r'Equacao de Laplace vs. Poisson — Funcao de Corrente Geostrofica ($\psi$)',
             fontsize=15, fontweight='bold', color=BLUE,
             fontfamily='serif', y=0.975)

plt.savefig('poisson_laplace_meteo.png', dpi=180,
            bbox_inches='tight', facecolor=BG)
print("Figura salva: poisson_laplace_meteo.png")
plt.show()
