"""
cfl_salva_png.py
================
Gera e salva figuras PNG do visualizador CFL para a equação de advecção 1D.

    ∂u/∂t + c ∂u/∂x = 0   —   Esquema FTBS (upwind)
    μ = c·Δt/Δx  (número de Courant)

USO — edite os parâmetros na seção "CONFIGURAÇÃO" abaixo e execute:

    python cfl_salva_png.py

Ou em Jupyter / JupyterLite:

    %run cfl_salva_png.py

Cada combinação de parâmetros gera um arquivo PNG separado com nome automático.
Não requer backend interativo — funciona em qualquer ambiente.

Dependências: numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # backend sem janela — funciona em todos os ambientes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO — edite aqui os cenários que deseja gerar
# ══════════════════════════════════════════════════════════════════════════════
# Cada linha é um cenário: (c, dt, N, steps)
#   c     = velocidade de advecção (m/s)
#   dt    = passo de tempo (s)
#   N     = número de pontos da grade espacial
#   steps = número de passos de tempo
CENARIOS = [
    (1.0, 0.009, 80,  20),   # μ ≈ 0.71  — estável, difusivo
    (1.0, 0.013, 80,  20),   # μ ≈ 1.03  — instável (limiar)
    (1.0, 0.016, 80,  20),   # μ ≈ 1.26  — instável, explode
    (2.0, 0.005, 80,  20),   # μ ≈ 0.79  — estável, c maior
    (1.0, 0.009, 120, 30),   # μ ≈ 1.08  — instável com grade fina
    (1.0, 0.006, 120, 30),   # μ ≈ 0.72  — estável com grade fina
]

PASTA_SAIDA = './'           # pasta onde os PNGs serão salvos
DPI         = 150            # resolução dos PNGs

# ══════════════════════════════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════════════════════════════
C_STABLE   = '#1baf7a'
C_UNSTABLE = '#e34948'
C_EXACT    = '#2a78d6'
C_INIT     = '#b4b2a9'
C_DIAG     = '#eda100'
C_BG_S     = '#d6f5e9'
C_BG_U     = '#fde8e8'

# ══════════════════════════════════════════════════════════════════════════════
# SIMULAÇÃO FTBS
# ══════════════════════════════════════════════════════════════════════════════
def gaussian(x, x0=0.2, sigma2=200):
    return np.exp(-sigma2 * (x - x0)**2)

def run_ftbs(N, c, dt, steps):
    dx      = 1.0 / (N - 1)
    mu      = c * dt / dx
    x       = np.linspace(0, 1, N)
    u       = gaussian(x)
    u0      = u.copy()
    for _ in range(steps):
        u[1:] = u[1:] - mu * (u[1:] - u[:-1])
    u       = np.clip(u, -4, 4)
    u_exact = gaussian(x, x0=0.2 + c * steps * dt)
    return x, u, u0, u_exact, dx, mu

# ══════════════════════════════════════════════════════════════════════════════
# DESENHO DO DIAGRAMA ESPAÇO-TEMPO (Fig. 4.3)
# ══════════════════════════════════════════════════════════════════════════════
def draw_book(ax, mu, stable):
    ax.cla()
    ax.set_facecolor('#f8f8f7')
    cols, rows = 5, 4

    diag_x = min(mu * rows, cols)

    # regiões coloridas
    if diag_x < cols:
        ax.add_patch(plt.Polygon([(0,0),(cols,0),(cols,rows),(diag_x,rows)],
                                  fc=C_BG_S, ec='none', zorder=0))
        ax.add_patch(plt.Polygon([(0,0),(diag_x,rows),(0,rows)],
                                  fc=C_BG_U, ec='none', zorder=0))
    else:
        tr = cols / mu
        ax.add_patch(plt.Polygon([(0,0),(cols,0),(cols,tr)],
                                  fc=C_BG_S, ec='none', zorder=0))
        ax.add_patch(plt.Polygon([(0,0),(cols,tr),(cols,rows),(0,rows)],
                                  fc=C_BG_U, ec='none', zorder=0))

    # grade pontilhada
    for ci in range(cols+1):
        ax.plot([ci,ci],[0,rows], lw=0.5, ls=':', color='#888', zorder=1)
    for ri in range(rows+1):
        ax.plot([0,cols],[ri,ri], lw=0.5, ls=':', color='#888', zorder=1)

    # diagonal CFL
    xe = min(diag_x, cols)
    ye = min(rows, cols/mu) if mu > 0 else rows
    ax.plot([0, xe],[0, ye], lw=2.5, color=C_DIAG, zorder=3)

    # pontos da grade
    for ri in [1,2,3]:
        for ci in [1,2,3]:
            ax.plot(ci+0.5, ri+0.5, 'o', ms=4, color='#444', zorder=4)

    # seta de propagação: (j,n) → (j+μ, n+1)
    mu_c    = min(mu, cols-2.5)
    arr_col = C_STABLE if stable else C_UNSTABLE
    ax.annotate('', xy=(2.5+mu_c, 3.5), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=arr_col,
                                lw=1.8, linestyle='dashed'))

    # rótulos de tempo: n-1 embaixo → n+1 em cima
    for ri, lbl in zip([1,2,3], ['n-1','n','n+1']):
        ax.text(-0.15, ri+0.5, lbl, ha='right', va='center',
                fontsize=9.5, color='#1a1a19')
    ax.text(-0.15, 0.5, 'Δt', ha='right', va='center',
            fontsize=8, color='#52514e')

    # rótulos de espaço
    for ci, lbl in zip([1,2,3], ['j-1','j','j+1']):
        ax.text(ci+0.5, -0.3, lbl, ha='center', va='top',
                fontsize=9.5, color='#1a1a19')
    ax.text(0.5, -0.3, 'Δx', ha='center', va='top',
            fontsize=8, color='#52514e')

    # rótulos das regiões
    ax.text(3.8, 1.0, 'Estável\nc < Δx/Δt',
            ha='center', fontsize=8.5, color='#0b7a53', fontweight='bold')
    ax.text(0.6, 3.5, 'Instável\nc > Δx/Δt',
            ha='center', fontsize=8.5, color='#a32d2d', fontweight='bold')

    ax.set_xlim(-0.4, cols+0.1); ax.set_ylim(-0.6, rows+0.3)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('t', fontsize=10, rotation=0, labelpad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    ax.set_title('Diagrama espaço-tempo (Fig. 4.3)', fontsize=10, pad=4)

    ax.annotate('', xy=(cols+0.1, 0), xytext=(-0.35, 0),
                arrowprops=dict(arrowstyle='->', color='#1a1a19', lw=1.2))
    ax.annotate('', xy=(0, rows+0.3), xytext=(0, -0.5),
                arrowprops=dict(arrowstyle='->', color='#1a1a19', lw=1.2))

# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAMA DE PARÂMETROS Δx × Δt
# ══════════════════════════════════════════════════════════════════════════════
def draw_param(ax, dx, dt, c):
    ax.cla()
    ax.set_facecolor('#f8f8f7')
    xmax, ymax = 0.028, 0.024
    x_int = ymax * c

    # regiões
    if x_int <= xmax:
        ax.fill_between([0, x_int, xmax], [0, ymax, ymax], 0,
                        color=C_BG_S, alpha=0.7, zorder=0)
        ax.fill_between([0, x_int], [0, ymax], ymax,
                        color=C_BG_U, alpha=0.7, zorder=0)
    else:
        tr = xmax / c
        ax.fill_between([0, xmax], [0, tr], 0,
                        color=C_BG_S, alpha=0.7, zorder=0)
        ax.fill_between([0, xmax], [0, tr], ymax,
                        color=C_BG_U, alpha=0.7, zorder=0)

    # diagonal μ=1
    x_d = np.array([0, min(x_int, xmax)])
    y_d = np.clip(x_d / c, 0, ymax)
    ax.plot(x_d, y_d, lw=2, color=C_DIAG, label='μ = 1')

    # ponto atual
    mu     = c * dt / dx
    stable = mu <= 1.0
    ax.plot(dx, dt, 'o', ms=9,
            color=C_STABLE if stable else C_UNSTABLE,
            mec='white', mew=1.5, zorder=5)

    # labels
    ax.text(0.022, 0.003, 'ESTÁVEL',  ha='center', fontsize=8.5,
            color='#0b7a53', fontweight='bold')
    ax.text(0.005, 0.020, 'INSTÁVEL', ha='center', fontsize=8.5,
            color='#a32d2d', fontweight='bold')
    ax.text(min(x_int, xmax)+0.0005, min(ymax, xmax/c)-0.001,
            'μ=1', fontsize=8, color=C_DIAG)

    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.set_xlabel('Δx', fontsize=10); ax.set_ylabel('Δt', fontsize=10)
    ax.set_title('Espaço de parâmetros (Δx × Δt)', fontsize=10, pad=4)
    ax.grid(True, lw=0.4, color='#d3d1c7')
    ax.legend(fontsize=9, loc='upper left')

# ══════════════════════════════════════════════════════════════════════════════
# GERAÇÃO DAS FIGURAS
# ══════════════════════════════════════════════════════════════════════════════
def gerar_figura(c, dt, N, steps):
    x, u, u0, u_exact, dx, mu = run_ftbs(N, c, dt, steps)
    stable = mu <= 1.0
    status = 'ESTAVEL' if stable else 'INSTAVEL'
    status_label = '✓ ESTÁVEL' if stable else '✗ INSTÁVEL'
    col_st = C_STABLE if stable else C_UNSTABLE

    fig = plt.figure(figsize=(13, 8), facecolor='#f8f8f7')
    fig.suptitle('Critério CFL — Equação de Advecção 1D (FTBS/Upwind)',
                 fontsize=13, fontweight='bold', y=0.99, color='#1a1a19')

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.10,
                           hspace=0.38, wspace=0.32)

    ax_sol  = fig.add_subplot(gs[0, :])
    ax_book = fig.add_subplot(gs[1, 0])
    ax_par  = fig.add_subplot(gs[1, 1])

    # ── painel 1: solução ────────────────────────────────────────────────────
    ax_sol.plot(x, u0,      color=C_INIT,   lw=1.5, ls='--', label='Condição inicial')
    ax_sol.plot(x, u_exact, color=C_EXACT,  lw=1.5, ls='-.', label='Solução exata')
    ax_sol.plot(x, u,       color=col_st,   lw=2.0,           label='FTBS (numérica)')
    ax_sol.set_xlim(0, 1); ax_sol.set_ylim(-1.6, 1.4)
    ax_sol.set_xlabel('x', fontsize=10); ax_sol.set_ylabel('u(x,t)', fontsize=10)
    ax_sol.legend(fontsize=9, loc='upper right', framealpha=0.8)
    ax_sol.grid(True, lw=0.4, color='#d3d1c7')
    ax_sol.axhline(0, color='#b4b2a9', lw=0.5)
    ax_sol.set_title(
        f'Solução numérica vs. exata   |   '
        f'Δx={dx:.4f}   Δt={dt:.3f}   μ={mu:.3f}   {status_label}   '
        f't={steps*dt:.3f} s',
        fontsize=9.5, pad=4, color='#1a1a19')

    # ── painel 2: diagrama do livro ──────────────────────────────────────────
    draw_book(ax_book, mu, stable)

    # ── painel 3: diagrama parâmetros ────────────────────────────────────────
    draw_param(ax_par, dx, dt, c)

    # ── barra de métricas ────────────────────────────────────────────────────
    msg = (f'Δx = {dx:.5f}   |   Δt = {dt:.3f} s   |   '
           f'μ = c·Δt/Δx = {mu:.3f}   |   '
           f'c = {c} m/s   |   N = {N}   |   passos = {steps}   |   '
           f'{"ESTÁVEL  (μ ≤ 1)" if stable else "INSTÁVEL  (μ > 1) — solução diverge!"}')
    fig.text(0.5, 0.05, msg, ha='center', va='top',
             fontsize=10, fontweight='bold', color=col_st,
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='#ffffff', edgecolor=col_st, lw=1.2))

    # ── rodapé ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
        'Equação de advecção 1D:  ∂u/∂t + c·∂u/∂x = 0   |   '
        'Esquema FTBS:  u_j^{n+1} = u_j^n − μ(u_j^n − u_{j−1}^n)',
        ha='center', fontsize=8, color='#52514e',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0efec',
                  edgecolor='#d3d1c7', lw=0.5))

    # ── salvar ───────────────────────────────────────────────────────────────
    nome = (f'{PASTA_SAIDA}cfl_c{c:.1f}_dt{dt:.4f}_N{N:03d}'
            f'_s{steps:02d}_mu{mu:.3f}_{status}.png')
    fig.savefig(nome, dpi=DPI, bbox_inches='tight', facecolor='#f8f8f7')
    plt.close(fig)
    print(f'  Salvo: {nome}')
    return nome

# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'\nGerando {len(CENARIOS)} figura(s)...\n')
    arquivos = []
    for (c, dt, N, steps) in CENARIOS:
        dx = 1.0 / (N - 1)
        mu = c * dt / dx
        print(f'  c={c}  dt={dt}  N={N}  steps={steps}  →  μ={mu:.3f}', end='  ')
        arq = gerar_figura(c, dt, N, steps)
        arquivos.append(arq)

    print(f'\nConcluído! {len(arquivos)} PNG(s) gerado(s) em "{PASTA_SAIDA}"')
