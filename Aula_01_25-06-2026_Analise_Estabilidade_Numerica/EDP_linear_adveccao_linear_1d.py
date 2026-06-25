"""
=============================================================================
Aula 01 — 25/06/2026
Advecção Linear 1D — Esquemas Forward (CTCS) e Upstream (FTBS)
=============================================================================
Conversão do programa Fortran (3 módulos + PROGRAM Main) com correções de bugs.

EDP de advecção linear 1D:
    ∂φ/∂t + u ∂φ/∂x = 0

Grade espacial:
    x(j) = 6·j·Δx,   j = 1 … iMax

Condição inicial correta:
    φ(x(j), 0) = sin²(x(j)) = sin²(6·j·Δx)

    BUG no Fortran (InitNumericalScheme): xl começa em 0.0 e só é
    atualizado APÓS o cálculo → PHI_C(1)=sin(0)²=0 em vez de sin(6·Δx)².
    Corrigido: xl = (6*i)*DX é calculado ANTES de PHI_C(i).

Solução analítica correta:
    φ_a(x, t) = sin²(x − u·t)   com  x = 6·j·Δx,  t = tn·Δt

    BUG no Fortran (AnaliticFunction): x(j) é lido ANTES de ser
    atualizado no loop → na primeira chamada todos x(j)=0; nos passos
    seguintes usa a grade do passo anterior defasada.
    Corrigido: x(j) = 6·j·Δx fixo, calculado ANTES de PHI_A(j).

Esquema Forward (centrado no espaço — CTCS):
    φ(j,n+1) = φ(j,n) − (u·Δt / 2·Δx) · [φ(j+1,n) − φ(j-1,n)]

Esquema Upstream (FTBS — Forward Time Backward Space):
    φ(j,n+1) = φ(j,n) − (u·Δt / Δx) · [φ(j,n) − φ(j-1,n)]

Parâmetros (idênticos ao Fortran):
    xdim        = 100           pontos de grade
    LX          = 1.0  m        comprimento do domínio
    Uvel        = 10.0 m/s      velocidade de advecção
    dx          = LX / xdim    espaçamento
    dt          = 0.1·dx/Uvel  passo de tempo  (CFL = 0.1 < 1)
    ninteraction= 20000         passos de tempo
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------------
# Parâmetros globais (idênticos ao PROGRAM Main do Fortran)
# ---------------------------------------------------------------------------
XDIM         = 100
LX           = 1.0          # m
UVEL         = 10.0         # m/s
DX           = LX / XDIM   # m
DT           = 0.1 * DX / UVEL   # s   →  CFL = u·Δt/Δx = 0.1
N_STEPS      = 20_000       # ninteraction


# ===========================================================================
# Módulo Class_Fields — campos alocados com halo de 2 pontos (índices -1 a iMax+2)
# ===========================================================================
class Fields:
    """
    Equivalente ao MODULE Class_Fields.
    Arrays com halo: índice 0 da array Python ↔ índice -1 do Fortran.
    Offset = 1  (Python index = Fortran index + 1)
    """
    OFFSET = 1   # PHI_fortran[-1] → PHI_python[0]

    def __init__(self, xdim: int, uvel: float):
        self.iMax = xdim
        self.Uvel = uvel
        size = xdim + 4            # índices -1 … iMax+2  →  tamanho = iMax + 4
        self.PHI_C = np.zeros(size)
        self.PHI_P = np.zeros(size)
        self.PHI_M = np.zeros(size)
        self.PHI_A = np.zeros(size)
        self.x     = np.zeros(size)

    def idx(self, i: int) -> int:
        """Converte índice Fortran → índice Python."""
        return i + self.OFFSET


# ===========================================================================
# Módulo Class_NumericalMethod
# ===========================================================================
class NumericalMethod:
    """
    Equivalente ao MODULE Class_NumericalMethod.
    """

    def __init__(self, fields: Fields, dt: float, dx: float):
        self.f  = fields
        self.Dt = dt
        self.Dx = dx
        self._init_condition()

    # -----------------------------------------------------------------------
    def _init_condition(self):
        """
        Equivalente à SUBROUTINE InitNumericalScheme — com correção de bug.

        BUG original no Fortran:
            xl = 0.0
            DO i = 1, iMax
                PHI_C(i) = sin(xl)**2   ← usa xl do passo ANTERIOR
                xl = (6*i) * DX         ← atualiza só depois
            END DO
            → PHI_C(1) = sin(0)²  = 0  (errado)
            → PHI_C(2) = sin(6·DX)²    (deveria ser sin(12·DX)²)

        Correção: xl = (6*i)*DX calculado ANTES de PHI_C(i).
            → PHI_C(j) = sin(6·j·Δx)²   para j = 1 … iMax
        """
        f = self.f
        for i in range(1, f.iMax + 1):
            xl = (6 * i) * self.Dx          # grade correta: x(j) = 6·j·Δx
            f.PHI_C[f.idx(i)] = np.sin(xl) ** 2

        f.PHI_M[:] = f.PHI_C
        f.PHI_P[:] = f.PHI_C

    # -----------------------------------------------------------------------
    def analytic_function(self, tn: int):
        """
        Equivalente à FUNCTION AnaliticFunction(tn) — com correção de bug.

        BUG original no Fortran:
            DO j = 1, iMax
                PHI_A(j) = sin( x(j) - u*tn*dt/(6*j*dx) )**2   ← x(j) ainda é 0 (ou do passo anterior)
                x(j) = (6*j)*DX                                  ← atualiza só depois
            END DO
            → No passo tn=0: x(j)=0 para todo j → PHI_A(j) = sin(-u*tn*dt/(6*j*dx))²
            → Nos demais passos: usa a grade calculada no passo anterior (defasagem de 1 passo)

        Correção: a grade é fixa, x(j) = 6·j·Δx, e deve ser calculada
        ANTES de avaliar PHI_A(j).

        Solução analítica correta:
            φ_a(j, tn) = sin²( x(j) − u·tn·Δt )
                       = sin²( 6·j·Δx − u·tn·Δt )
        """
        f = self.f
        for j in range(1, f.iMax + 1):
            xj  = (6 * j) * self.Dx               # grade fixa
            t   = tn * self.Dt                     # tempo físico
            f.PHI_A[f.idx(j)] = np.sin(xj - f.Uvel * t) ** 2

    # -----------------------------------------------------------------------
    def scheme_forward(self):
        """
        Esquema CTCS — centrado no espaço (Forward no tempo):
            φ(j,n+1) = φ(j,n) − (u·Δt / 2·Δx) · [φ(j+1,n) − φ(j-1,n)]

        Nota: no Fortran, SchemeForward usa  Uvel*Dt/2*Dx  (sem parênteses
        em torno de 2*Dx), o que equivale a  (Uvel*Dt/2)*Dx.
        Aqui preservamos a mesma precedência:  coef = Uvel*Dt/(2) * Dx
        → para fins didáticos, interpretamos como erro tipográfico e usamos
          o coeficiente correto  Uvel*Dt / (2*Dx).
        """
        f   = self.f
        cfl = f.Uvel * self.Dt / (2.0 * self.Dx)
        for j in range(1, f.iMax + 1):
            f.PHI_P[f.idx(j)] = (f.PHI_C[f.idx(j)]
                                  - cfl * (f.PHI_C[f.idx(j+1)] - f.PHI_C[f.idx(j-1)]))
        self._update_boundary()

    # -----------------------------------------------------------------------
    def scheme_upstream(self):
        """
        Esquema FTBS — Upstream (Forward no tempo, Backward no espaço):
            φ(j,n+1) = φ(j,n) − (u·Δt / Δx) · [φ(j,n) − φ(j-1,n)]
        """
        f   = self.f
        cfl = f.Uvel * self.Dt / self.Dx
        for j in range(1, f.iMax + 1):
            f.PHI_P[f.idx(j)] = (f.PHI_C[f.idx(j)]
                                  - cfl * (f.PHI_C[f.idx(j)] - f.PHI_C[f.idx(j-1)]))
        self._update_boundary()

    # -----------------------------------------------------------------------
    def _update_boundary(self):
        """
        Equivalente à SUBROUTINE UpdateBoundaryLayer.
        Condição de contorno periódica nos halos.
        """
        f = self.f
        f.PHI_P[f.idx( 0      )] = f.PHI_P[f.idx(f.iMax    )]
        f.PHI_P[f.idx(-1      )] = f.PHI_P[f.idx(f.iMax - 1)]
        f.PHI_P[f.idx(f.iMax+1)] = f.PHI_P[f.idx(1          )]
        f.PHI_P[f.idx(f.iMax+2)] = f.PHI_P[f.idx(2          )]

    # -----------------------------------------------------------------------
    def scheme_update(self):
        """
        Equivalente à FUNCTION SchemeUpdate:
            PHI_M ← PHI_C
            PHI_C ← PHI_P
        """
        f = self.f
        f.PHI_M[:] = f.PHI_C
        f.PHI_C[:] = f.PHI_P


# ===========================================================================
# Simulação principal — equivalente à SUBROUTINE Run
# ===========================================================================
def run_simulation(scheme: str = "forward", n_steps: int = N_STEPS,
                   snapshot_steps: list = None) -> dict:
    """
    Executa a simulação de advecção linear 1D.

    Parâmetros
    ----------
    scheme         : "forward" (CTCS) ou "upstream" (FTBS)
    n_steps        : número de iterações
    snapshot_steps : lista de passos de tempo para salvar snapshots

    Retorna dicionário com snapshots de PHI_C e PHI_A.
    """
    if snapshot_steps is None:
        snapshot_steps = [0, 100, 500, 2000, 5000, n_steps]

    fields = Fields(XDIM, UVEL)
    nm     = NumericalMethod(fields, DT, DX)

    # grade espacial interna (índices 1…iMax do Fortran)
    j_idx = np.arange(1, XDIM + 1)
    x_grid = (6 * j_idx) * DX    # igual ao Fortran: x(j) = 6*j*DX

    snaps_num  = {}
    snaps_ana  = {}

    for tn in range(0, n_steps + 1):
        if scheme == "forward":
            nm.scheme_forward()
        else:
            nm.scheme_upstream()

        nm.analytic_function(tn)

        if tn in snapshot_steps:
            snaps_num[tn] = fields.PHI_C[[fields.idx(j) for j in j_idx]].copy()
            snaps_ana[tn] = fields.PHI_A[[fields.idx(j) for j in j_idx]].copy()

        nm.scheme_update()

    return {
        "scheme"    : scheme,
        "x"         : x_grid,
        "snaps_num" : snaps_num,
        "snaps_ana" : snaps_ana,
        "fields"    : fields,
        "n_steps"   : n_steps,
        "cfl"       : UVEL * DT / DX,
    }


# ===========================================================================
# Visualizações didáticas
# ===========================================================================
def plot_results(sim_fwd: dict, sim_ups: dict):
    """
    Quatro painéis didáticos comparando Forward (CTCS) e Upstream (FTBS).
    """
    C_ANA   = "#2c3e50"
    C_FWD   = "#2980b9"
    C_UPS   = "#e74c3c"
    C_INIT  = "#7f8c8d"

    snap_steps = sorted(sim_fwd["snaps_num"].keys())
    x = sim_fwd["x"]

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("#f7f9fc")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # =========================================================
    # Painel 1 — Evolução temporal: Forward (CTCS)
    # =========================================================
    ax1 = fig.add_subplot(gs[0, 0])
    alphas = np.linspace(0.25, 1.0, len(snap_steps))
    for i, tn in enumerate(snap_steps):
        lbl = f"tn={tn}"
        ax1.plot(x, sim_fwd["snaps_num"][tn], color=C_FWD, alpha=alphas[i],
                 lw=1.6, label=lbl)
    ax1.plot(x, sim_fwd["snaps_ana"][snap_steps[-1]], color=C_ANA,
             lw=2.0, ls="--", label=f"Analítica (tn={snap_steps[-1]})")
    ax1.set_title(f"Esquema CTCS (Forward centrado)\nCFL = {sim_fwd['cfl']:.2f}",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("x [m]", fontsize=10)
    ax1.set_ylabel("φ(x, t)", fontsize=10)
    ax1.legend(fontsize=7.5, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#fdfdfd")

    # =========================================================
    # Painel 2 — Evolução temporal: Upstream (FTBS)
    # =========================================================
    ax2 = fig.add_subplot(gs[0, 1])
    for i, tn in enumerate(snap_steps):
        ax2.plot(x, sim_ups["snaps_num"][tn], color=C_UPS, alpha=alphas[i],
                 lw=1.6, label=f"tn={tn}")
    ax2.plot(x, sim_ups["snaps_ana"][snap_steps[-1]], color=C_ANA,
             lw=2.0, ls="--", label=f"Analítica (tn={snap_steps[-1]})")
    ax2.set_title(f"Esquema FTBS (Upstream)\nCFL = {sim_ups['cfl']:.2f}",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("x [m]", fontsize=10)
    ax2.set_ylabel("φ(x, t)", fontsize=10)
    ax2.legend(fontsize=7.5, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#fdfdfd")

    # =========================================================
    # Painel 3 — Comparação no passo final: CTCS vs FTBS vs Analítica
    # =========================================================
    ax3 = fig.add_subplot(gs[1, 0])
    tn_final = snap_steps[-1]
    ax3.plot(x, sim_fwd["snaps_ana"][tn_final], color=C_ANA, lw=2.5,
             label="Analítica", zorder=4)
    ax3.plot(x, sim_fwd["snaps_num"][tn_final], color=C_FWD, lw=2.0,
             ls="--", label=f"CTCS (Forward)", zorder=3)
    ax3.plot(x, sim_ups["snaps_num"][tn_final], color=C_UPS, lw=2.0,
             ls="-.", label=f"FTBS (Upstream)", zorder=3)

    ax3.set_title(f"Comparação no passo tn = {tn_final}\n(CTCS vs FTBS vs Analítica)",
                  fontsize=12, fontweight="bold")
    ax3.set_xlabel("x [m]", fontsize=10)
    ax3.set_ylabel("φ(x, t)", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#fdfdfd")

    # Anotação: características de cada esquema
    ax3.text(0.02, 0.97,
             "CTCS: dispersivo, sem dissipação\nFTBS: difusivo, estável para u > 0",
             transform=ax3.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#95a5a6", alpha=0.9))

    # =========================================================
    # Painel 4 — Erro RMS ao longo do tempo
    # =========================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Re-simular com mais snapshots para curva de erro suave
    err_steps = list(range(0, N_STEPS + 1, 200))
    f_err = Fields(XDIM, UVEL)
    u_err = Fields(XDIM, UVEL)
    nm_f  = NumericalMethod(f_err, DT, DX)
    nm_u  = NumericalMethod(u_err, DT, DX)

    j_idx = np.arange(1, XDIM + 1)
    rms_fwd = []
    rms_ups = []
    t_err   = []

    for tn in range(0, N_STEPS + 1):
        nm_f.scheme_forward()
        nm_u.scheme_upstream()
        nm_f.analytic_function(tn)
        nm_u.analytic_function(tn)

        if tn % 200 == 0:
            phi_c_f = f_err.PHI_C[[f_err.idx(j) for j in j_idx]]
            phi_a_f = f_err.PHI_A[[f_err.idx(j) for j in j_idx]]
            phi_c_u = u_err.PHI_C[[u_err.idx(j) for j in j_idx]]
            phi_a_u = u_err.PHI_A[[u_err.idx(j) for j in j_idx]]

            rms_fwd.append(np.sqrt(np.mean((phi_c_f - phi_a_f)**2)))
            rms_ups.append(np.sqrt(np.mean((phi_c_u - phi_a_u)**2)))
            t_err.append(tn * DT)

        nm_f.scheme_update()
        nm_u.scheme_update()

    t_err   = np.array(t_err)
    rms_fwd = np.array(rms_fwd)
    rms_ups = np.array(rms_ups)

    ax4.plot(t_err, rms_fwd, color=C_FWD, lw=2.0, label="CTCS (Forward)")
    ax4.plot(t_err, rms_ups, color=C_UPS, lw=2.0, label="FTBS (Upstream)")
    ax4.set_title("Erro RMS ao longo do tempo\n(numérico vs analítico)",
                  fontsize=12, fontweight="bold")
    ax4.set_xlabel("Tempo [s]", fontsize=10)
    ax4.set_ylabel("RMSE  (φ_num − φ_ana)", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor("#fdfdfd")

    # =========================================================
    # Título geral
    # =========================================================
    fig.suptitle(
        "Aula 01 — Advecção Linear 1D\n"
        r"$\partial\phi/\partial t + u\,\partial\phi/\partial x = 0$"
        f"     |     u = {UVEL} m/s,  Δx = {DX:.4f} m,  "
        f"Δt = {DT:.5f} s,  CFL = {UVEL*DT/DX:.2f}",
        fontsize=13, fontweight="bold", y=0.99
    )

    out_path = "/mnt/user-data/outputs/adveccao_linear_1d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figura salva: adveccao_linear_1d.png")
    plt.show()


# ===========================================================================
# Execução principal — equivalente ao PROGRAM Main
# ===========================================================================
if __name__ == "__main__":

    cfl = UVEL * DT / DX

    print("=" * 62)
    print("  Advecção Linear 1D — CTCS e FTBS")
    print("=" * 62)
    print(f"  xdim         = {XDIM}")
    print(f"  LX           = {LX} m")
    print(f"  Uvel         = {UVEL} m/s")
    print(f"  dx           = {DX:.6f} m")
    print(f"  dt           = {DT:.6f} s")
    print(f"  CFL (u·dt/dx)= {cfl:.4f}  {'✓ < 1 estável' if cfl < 1 else '✗ ≥ 1 instável'}")
    print(f"  n_steps      = {N_STEPS}")
    print("=" * 62)

    snap_steps = [0, 100, 500, 2000, 5000, N_STEPS]

    print("\n  Rodando esquema Forward (CTCS)...")
    sim_fwd = run_simulation("forward",  N_STEPS, snap_steps)

    print("  Rodando esquema Upstream (FTBS)...")
    sim_ups = run_simulation("upstream", N_STEPS, snap_steps)

    print("\n  Gerando figuras diagnósticas...")
    plot_results(sim_fwd, sim_ups)
