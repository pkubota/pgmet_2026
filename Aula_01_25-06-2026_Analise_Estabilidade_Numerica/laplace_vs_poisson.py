"""
Laplace vs Poisson — visualização comparativa
=============================================
Dependências: numpy, scipy, matplotlib
Instalar:  pip install numpy scipy matplotlib

Autor: gerado para curso de Métodos Numéricos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# ── paleta ───────────────────────────────────────────────────────────────────
BG     = '#fdf6e3'
BLUE   = '#1a3a6e'
RED    = '#c0392b'
DGRAY  = '#555555'

# ─────────────────────────────────────────────────────────────────────────────
# 1. SOLVER: diferenças finitas 2-D, Dirichlet, domínio [0,1]²
# ─────────────────────────────────────────────────────────────────────────────
def solve_fd_2d(N=40, source_amplitude=0.0):
    """
    Resolve  ∇²u = f(x,y)  em [0,1]² por diferenças finitas centrais.

    Condições de contorno (Dirichlet):
        u = sin(π x)  na borda superior  (y = 1)
        u = 0         nas demais bordas

    Parâmetro
    ---------
    source_amplitude : float
        Amplitude de  f(x,y) = A · sin(πx) · sin(πy).
        0  → equação de Laplace
        ≠0 → equação de Poisson

    Retorna
    -------
    X, Y : arrays (N+2, N+2)  — coordenadas da malha incluindo fronteiras
    U    : array  (N+2, N+2)  — solução
    """
    h = 1.0 / (N + 1)
    x = np.linspace(0, 1, N + 2)
    y = np.linspace(0, 1, N + 2)
    X, Y = np.meshgrid(x, y)

    size = N * N
    A_mat = np.zeros((size, size))
    b_vec = np.zeros(size)

    def idx(i, j):
        return i * N + j          # índice vetorial do nó interior (i,j)

    for i in range(N):            # i → direção y (linha)
        for j in range(N):        # j → direção x (coluna)
            xi = x[j + 1]
            yi = y[i + 1]
            k  = idx(i, j)

            A_mat[k, k] = -4.0   # coeficiente central do stencil

            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    A_mat[k, idx(ni, nj)] = 1.0    # vizinho interior
                else:
                    # vizinho é fronteira → contribui ao lado direito
                    bxi = x[nj + 1] if 0 <= nj < N else (0.0 if nj < 0 else 1.0)
                    byi = y[ni + 1] if 0 <= ni < N else (0.0 if ni < 0 else 1.0)
                    u_bc = np.sin(np.pi * bxi) if (ni >= N) else 0.0
                    b_vec[k] -= u_bc

            # termo fonte f(x,y) = A · sin(πx) · sin(πy)
            b_vec[k] += h**2 * source_amplitude * np.sin(np.pi * xi) * np.sin(np.pi * yi)

    u_inner = np.linalg.solve(A_mat, b_vec)

    U = np.zeros((N + 2, N + 2))
    for i in range(N):
        for j in range(N):
            U[i + 1, j + 1] = u_inner[idx(i, j)]
    U[-1, :] = np.sin(np.pi * x)   # borda superior quente
    return X, Y, U


# ─────────────────────────────────────────────────────────────────────────────
# 2. SOLUÇÃO ANALÍTICA — condução de calor
#    Laplace: T = sin(πx)·sinh(πy)/sinh(π)   [série de Fourier, 1 termo]
#    Poisson: soma do Laplace + particular para f = −Q/k·sin(πx)·sin(πy)
# ─────────────────────────────────────────────────────────────────────────────
def analitica_laplace(x, y):
    """Solução exata de ∇²T = 0 com T(x,1)=sin(πx), demais=0."""
    return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)

def analitica_poisson(x, y, Q_k=5.0):
    """
    Solução aproximada de ∇²T = -Q/k·sin(πx)·sin(πy) com mesmas BCs.
    Particular: T_p = (Q/k)/(2π²) · sin(πx)·sin(πy)
    """
    T_homog = analitica_laplace(x, y)
    T_part  = (Q_k / (2 * np.pi**2)) * np.sin(np.pi * x) * np.sin(np.pi * y)
    return T_homog + T_part


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURA
# ─────────────────────────────────────────────────────────────────────────────
def plotar():
    # --- resolve numericamente ---
    N = 40
    X_n, Y_n, U_lap = solve_fd_2d(N, source_amplitude=0.0)
    X_n, Y_n, U_poi = solve_fd_2d(N, source_amplitude=8.0)

    # --- solução analítica (malha fina) ---
    xf = np.linspace(0, 1, 200)
    yf = np.linspace(0, 1, 200)
    Xf, Yf = np.meshgrid(xf, yf)
    T_lap_an = analitica_laplace(Xf, Yf)
    T_poi_an = analitica_poisson(Xf, Yf, Q_k=5.0)

    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            hspace=0.55, wspace=0.38,
                            top=0.95, bottom=0.04,
                            left=0.07, right=0.97)

    # ── painel 0: equações ───────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(BG)
    ax0.axis('off')

    ax0.add_patch(Rectangle((0.02, 0.08), 0.43, 0.84,
        transform=ax0.transAxes, facecolor='#e8f0fb', edgecolor=BLUE, lw=2))
    ax0.text(0.235, 0.82, 'Equacao de Laplace',
             transform=ax0.transAxes, ha='center', fontsize=14,
             color=BLUE, fontweight='bold', fontfamily='serif')
    ax0.text(0.235, 0.50, r'$\nabla^2 u = 0$',
             transform=ax0.transAxes, ha='center', fontsize=24, color=BLUE)
    ax0.text(0.235, 0.18,
             'Sem fontes internas\nMinimos/maximos apenas na fronteira\n(Principio do Maximo)',
             transform=ax0.transAxes, ha='center', fontsize=10,
             color=BLUE, style='italic')

    ax0.add_patch(Rectangle((0.55, 0.08), 0.43, 0.84,
        transform=ax0.transAxes, facecolor='#fdecea', edgecolor=RED, lw=2))
    ax0.text(0.765, 0.82, 'Equacao de Poisson',
             transform=ax0.transAxes, ha='center', fontsize=14,
             color=RED, fontweight='bold', fontfamily='serif')
    ax0.text(0.765, 0.50, r'$\nabla^2 u = f(x,y)$',
             transform=ax0.transAxes, ha='center', fontsize=24, color=RED)
    ax0.text(0.765, 0.18,
             r'$f \neq 0$: fonte ou sumidouro pertuba o campo' + '\n'
             r'Maximos podem ocorrer no interior do dominio',
             transform=ax0.transAxes, ha='center', fontsize=10,
             color=RED, style='italic')

    # ── painel 1-2: solucao numerica ─────────────────────────────────────────
    kw = dict(cmap='RdYlBu_r', shading='auto')

    ax1 = fig.add_subplot(gs[1, 0]); ax1.set_facecolor(BG)
    cf1 = ax1.pcolormesh(X_n, Y_n, U_lap, **kw, vmin=0, vmax=1)
    ax1.contour(X_n, Y_n, U_lap, levels=10, colors='k', linewidths=0.6, alpha=0.5)
    plt.colorbar(cf1, ax=ax1, shrink=0.85, label='u')
    ax1.set_title(r'$\nabla^2 u = 0$  [num. FD]', fontsize=12, color=BLUE, fontweight='bold')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.text(0.5, -0.05, 'BC: u = sin(px) no topo, u = 0 demais',
             transform=ax1.transAxes, ha='center', fontsize=8.5,
             color=DGRAY, style='italic')

    ax2 = fig.add_subplot(gs[1, 1]); ax2.set_facecolor(BG)
    cf2 = ax2.pcolormesh(X_n, Y_n, U_poi, **kw)
    ax2.contour(X_n, Y_n, U_poi, levels=10, colors='k', linewidths=0.6, alpha=0.5)
    plt.colorbar(cf2, ax=ax2, shrink=0.85, label='u')
    ax2.set_title(r'$\nabla^2 u = 8\sin(\pi x)\sin(\pi y)$  [num. FD]',
                  fontsize=12, color=RED, fontweight='bold')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.text(0.5, -0.05, r'Mesmas BCs + fonte distribuida no interior',
             transform=ax2.transAxes, ha='center', fontsize=8.5,
             color=DGRAY, style='italic')

    # ── painel 3-4: solucao analitica — conducao de calor ────────────────────
    ax3 = fig.add_subplot(gs[2, 0]); ax3.set_facecolor(BG)
    cf3 = ax3.contourf(Xf, Yf, T_lap_an, levels=20, cmap='hot_r', alpha=0.9)
    ax3.contour(Xf, Yf, T_lap_an, levels=10, colors='white', linewidths=0.8, alpha=0.6)
    plt.colorbar(cf3, ax=ax3, shrink=0.85, label='T (norm.)')
    ax3.set_title('[Laplace]  Conducao de calor estacionaria\n'
                  r'$\nabla^2 T = 0$  (sem geracao interna)',
                  fontsize=10.5, color=BLUE, fontweight='bold')
    ax3.set_xlabel('x'); ax3.set_ylabel('y')
    imax_l = np.unravel_index(T_lap_an.argmax(), T_lap_an.shape)
    ax3.text(0.5, 0.97,
             f'max T = {T_lap_an.max():.3f}  na fronteira superior',
             transform=ax3.transAxes, ha='center', va='top',
             fontsize=8.5, color='white',
             bbox=dict(facecolor=BLUE, alpha=0.6, boxstyle='round,pad=0.3'))

    ax4 = fig.add_subplot(gs[2, 1]); ax4.set_facecolor(BG)
    cf4 = ax4.contourf(Xf, Yf, T_poi_an, levels=20, cmap='hot_r', alpha=0.9)
    ax4.contour(Xf, Yf, T_poi_an, levels=10, colors='white', linewidths=0.8, alpha=0.6)
    plt.colorbar(cf4, ax=ax4, shrink=0.85, label='T (norm.)')
    ax4.set_title('[Poisson]  Conducao + geracao interna\n'
                  r'$\nabla^2 T = -Q/k$  (resistor, reator nuclear)',
                  fontsize=10.5, color=RED, fontweight='bold')
    ax4.set_xlabel('x'); ax4.set_ylabel('y')
    imax_p = np.unravel_index(T_poi_an.argmax(), T_poi_an.shape)
    ax4.plot(Xf[imax_p], Yf[imax_p], '*', color='yellow', ms=14, zorder=5)
    ax4.text(0.5, 0.97,
             f'max T = {T_poi_an.max():.3f}  no INTERIOR (x={Xf[imax_p]:.2f}, y={Yf[imax_p]:.2f})',
             transform=ax4.transAxes, ha='center', va='top',
             fontsize=8.5, color='white',
             bbox=dict(facecolor=RED, alpha=0.6, boxstyle='round,pad=0.3'))

    # ── painel 5: tabela de exemplos fisicos ──────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.set_facecolor(BG); ax5.axis('off')

    headers = ['Area', 'Laplace  (nabla^2 u = 0)', 'Poisson  (nabla^2 u = f)']
    rows = [
        ['Eletrostatica',
         'Potencial eletrico no vacuo\n(sem cargas livres)',
         'nabla^2 phi = -rho/eps0\n(lei de Gauss diferencial)'],
        ['Conducao\nde calor',
         'Temperatura estacionaria\nsem geracao interna',
         'Com geracao Q:  nabla^2 T = -Q/k\n(resistor, reator nuclear)'],
        ['Mecanica\nde fluidos',
         'Escoamento irrotacional\nnabla^2 phi = 0  (pot. de velocidade)',
         'Funcao de corrente:\nnabla^2 psi = -omega  (vorticidade)'],
        ['Meteorologia\n/ NWP',
         'Funcao de corrente em\nescoamento geostrof. sem vort. rel.',
         'Eq. de Poisson para pressao:\nnabla^2 p\' = f(div u)  nos modelos NWP'],
    ]

    col_x = [0.01,  0.145, 0.565]
    col_w = [0.125, 0.405, 0.415]
    row_h  = 0.185
    row_y0 = 0.93

    hdr_colors = ['#444444', BLUE, RED]
    for ci, (h, xc, w, hc) in enumerate(zip(headers, col_x, col_w, hdr_colors)):
        ax5.add_patch(Rectangle((xc, row_y0 - 0.075), w - 0.005, 0.08,
            transform=ax5.transAxes, facecolor=hc, edgecolor='white', lw=1))
        ax5.text(xc + (w - 0.005) / 2, row_y0 - 0.035, h,
                 transform=ax5.transAxes, ha='center', va='center',
                 fontsize=9.5, color='white', fontweight='bold')

    for ri, row in enumerate(rows):
        y = row_y0 - 0.075 - (ri + 1) * row_h
        for ci, (cell, xc, w) in enumerate(zip(row, col_x, col_w)):
            fc = ('#e8f0fb' if ri % 2 == 0 else '#dce8fa') if ci == 1 else \
                 ('#fdecea' if ri % 2 == 0 else '#fce0de') if ci == 2 else \
                 ('#f5f5f0' if ri % 2 == 0 else BG)
            ax5.add_patch(Rectangle((xc, y), w - 0.005, row_h - 0.01,
                transform=ax5.transAxes, facecolor=fc, edgecolor='#ccc', lw=0.7))
            cc = BLUE if ci == 1 else (RED if ci == 2 else '#333')
            ax5.text(xc + (w - 0.005) / 2, y + (row_h - 0.01) / 2, cell,
                     transform=ax5.transAxes, ha='center', va='center',
                     fontsize=8.5, color=cc, linespacing=1.4)

    ax5.set_title('Exemplos Fisicos', fontsize=11, color='#333',
                  fontweight='bold', loc='left', x=0.01)

    plt.suptitle('Equacao de Laplace  vs.  Equacao de Poisson',
                 fontsize=16, fontweight='bold', color=BLUE,
                 fontfamily='serif', y=0.975)

    plt.savefig('laplace_vs_poisson.png', dpi=180,
                bbox_inches='tight', facecolor=BG)
    print("Figura salva em: laplace_vs_poisson.png")
    plt.show()


if __name__ == '__main__':
    plotar()
