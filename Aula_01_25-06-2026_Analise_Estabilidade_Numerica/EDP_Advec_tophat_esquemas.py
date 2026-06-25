"""
=============================================================================
Aula 01 — 25/06/2026
Advecção Linear 1D — Comparação de Esquemas Numéricos
Pulso Top-Hat com Condições de Contorno Periódicas
=============================================================================
Conversão do programa Fortran (Class_Fields + ModAdvection + Main).

EDP de advecção linear 1D:
    ∂u/∂t + C ∂u/∂x = 0

Condição inicial:
    Pulso top-hat (chapéu) de amplitude Area ≈ 0.963
    entre xb0 = xa[Idim-1]/4  e  xf0 = xa[Idim-1]/2

Contorno: periódico (o pulso "sai" pela direita e "entra" pela esquerda)

Parâmetros (idênticos ao Fortran):
    Idim   = 100
    DeltaX = 1.0
    DeltaT = 1.55
    C      = 0.2   →  CFL = C·ΔT/ΔX = 0.31
    niter  = 300

Esquemas implementados (todos os 7 do Fortran):
  1. UpWind_1thBW    — FTBS, 1ª ordem, backward no espaço
  2. FTCS_2thCS      — centrado 2ª ordem no espaço
  3. FTCS_2thCS2     — experimental misto (ativo no Fortran comentado)
  4. FTCS_3thCS      — misto 3ª ordem
  5. FTCS_4thCS      — diferença 4ª ordem com parâmetro mi=0.7
  6. CTCS_2thCS      — leap-frog (preservado com o bug original)
  7. Estavel_4thCS   — expansão de Taylor 4ª ordem  ← ESQUEMA ATIVO

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Parâmetros globais (idênticos ao Fortran)
# ---------------------------------------------------------------------------
IDIM    = 100
DELTA_X = 1.0
DELTA_T = 1.55
C       = 0.2
NITER   = 300
CFL     = C * DELTA_T / DELTA_X   # = 0.31

# ===========================================================================
# Funções de indexação periódica — equivalentes às SUBROUTINEs index / index2
# Implementadas como arrays NumPy (vetorizadas para todos os i de 1…Idim)
# Fortran usa índice base-1; Python usa base-0.
# ===========================================================================

def make_index(idim: int):
    """
    Equivalente à SUBROUTINE index(i, xb, xc, xf).
    Retorna arrays de índices base-0 para vizinhos com CC periódica.
    """
    ic = np.arange(idim)                   # xc: 0 … idim-1
    ib = (ic - 1) % idim                   # xb: periódico
    if_ = (ic + 1) % idim                  # xf: periódico
    return ib, ic, if_


def make_index2(idim: int):
    """
    Equivalente à SUBROUTINE index2(i, xb3,xb2,xb, xc, xf,xf2,xf3).
    Retorna arrays de índices base-0 com halo de 3 pontos, CC periódica.
    """
    ic  = np.arange(idim)
    ib  = (ic - 1) % idim
    ib2 = (ic - 2) % idim
    ib3 = (ic - 3) % idim
    if_ = (ic + 1) % idim
    if2 = (ic + 2) % idim
    if3 = (ic + 3) % idim
    return ib3, ib2, ib, ic, if_, if2, if3


# Pré-calcular índices (usados em todos os esquemas)
IB, IC, IF_ = make_index(IDIM)
IB3, IB2, IB_4, IC_4, IF_4, IF2, IF3 = make_index2(IDIM)


# ===========================================================================
# Inicialização — equivalente à SUBROUTINE Init_Class_Fields
# ===========================================================================

def init_fields():
    """
    Constrói a grade, o pulso top-hat e calcula a Area.
    Retorna dict com xa, u, ua, xb0, xf0, tb0, tf0, xxb, yyf, Area.
    """
    xa = np.arange(IDIM, dtype=np.float64) * DELTA_X   # xa(i) = (i-1)*DX, base-0

    xb0 = xa[-1] / 4.0    # = 24.75
    xf0 = xa[-1] / 2.0    # = 49.5
    tb0 = 0.0
    tf0 = 0.0
    t   = 0.0

    xxb = xb0 + C * (t - tb0)   # = 24.75
    yyf = xf0 + C * (t - tf0)   # = 49.5

    # Pulso binário inicial
    u = np.where((xa > xxb) & (xa < yyf), 1.0, 0.0)

    # Cálculo de Area — fiel ao Fortran
    xb_idx = np.argmin(np.abs(xa - xxb))   # MINLOC(ABS(xa - xxb))
    xf_idx = np.argmin(np.abs(xa - yyf))   # MINLOC(ABS(xa - yyf))

    # Fortran: Area = (u(xf)-u(xb-1)) * (xa(xf)-xa(xb)) / (xf-xb+1)
    # Nota: xb-1 em base-1 = xb_idx-1 em base-0
    area = ((u[xf_idx] - u[xb_idx - 1]) *
            (xa[xf_idx] - xa[xb_idx]) /
            (xf_idx - xb_idx + 1))

    # Substitui 1.0 por Area
    u = np.where(u == 1.0, area, u)

    ua = u.copy()

    print(f"  Inicialização:")
    print(f"    xa: [{xa[0]:.1f} … {xa[-1]:.1f}]")
    print(f"    xb0={xb0:.2f}, xf0={xf0:.2f}")
    print(f"    xxb={xxb:.2f}, yyf={yyf:.2f}")
    print(f"    xb_idx={xb_idx}, xf_idx={xf_idx}")
    print(f"    Area = {area:.6f}")
    print(f"    Pulso ativo em {np.sum(u > 0)} pontos")

    return {
        "xa"   : xa,
        "u"    : u,
        "ua"   : ua,
        "xb0"  : xb0,
        "xf0"  : xf0,
        "tb0"  : tb0,
        "tf0"  : tf0,
        "xxb"  : xxb,
        "yyf"  : yyf,
        "area" : area,
    }


# ===========================================================================
# Solução analítica — equivalente à FUNCTION AnaliticFunction
# ===========================================================================

def analytic_function(state: dict, it: int) -> np.ndarray:
    """
    Pulso top-hat advectado com CC periódica manual.
    Atualiza state in-place (xb0, xf0, tb0, tf0, xxb, yyf).
    """
    xa   = state["xa"]
    area = state["area"]
    t    = it * DELTA_T

    # Atualiza borda dianteira (yyf)
    state["yyf"] = state["xf0"] + C * (t - state["tf0"])
    if state["yyf"] >= xa[-1]:
        state["xf0"] = 0.0
        state["yyf"] = state["xf0"]
        state["tf0"] = t

    # Atualiza borda traseira (xxb)
    state["xxb"] = state["xb0"] + C * (t - state["tb0"])
    if state["xxb"] >= xa[-1]:
        state["xb0"] = 0.0
        state["xxb"] = state["xb0"]
        state["tb0"] = t

    xxb = state["xxb"]
    yyf = state["yyf"]

    termXa = np.zeros(IDIM)

    if state["xf0"] <= state["xb0"] and yyf <= xxb:
        # Pulso "partido" — cruzou o contorno periódico
        termXa = np.where((xa > xxb) | (xa < yyf), area, 0.0)
    else:
        # Pulso contíguo normal
        termXa = np.where((xa > xxb) & (xa < yyf), area, 0.0)

    return termXa


# ===========================================================================
# Esquemas numéricos — equivalentes às FUNCTIONs do MODULE ModAdvection
# ===========================================================================

def scheme_upwind_1th(u: np.ndarray) -> np.ndarray:
    """
    FTBS — Upstream 1ª ordem backward no espaço:
        u(i,n+1) = u(i,n) - (C·ΔT/ΔX)·[u(i,n) - u(i-1,n)]
    """
    return u[IC] - (C * DELTA_T / DELTA_X) * (u[IC] - u[IB])


def scheme_ftcs_2th(u: np.ndarray) -> np.ndarray:
    """
    FTCS — centrado 2ª ordem no espaço:
        u(i,n+1) = u(i,n) - C·ΔT/(2·ΔX)·[u(i+1,n) - u(i-1,n)]
    """
    return u[IC] - (C * DELTA_T / (2.0 * DELTA_X)) * (u[IF_] - u[IB])


def scheme_ftcs_2th2(u: np.ndarray) -> np.ndarray:
    """
    Esquema misto experimental (Solve_Estabilidade_FTCS_2thCS2).
    Fórmula ativa no Fortran (última antes de END DO):
        u(i,n+1) = u(i,n) - C·ΔT·[
            (u(i+2) - 9·u(i-1) + 8·u(i-1))/(6·ΔX)
          + (u(i+2) - 2·u(i) + u(i-2))/(4·ΔX)
        ]
    Preservado fielmente, incluindo possível erro tipográfico do original.
    """
    return u[IC] - (C * DELTA_T) * (
        (u[IF2] - 9*u[IB_4] + 8*u[IB_4]) / (6.0 * DELTA_X) +
        (u[IF2] - 2*u[IC_4] + u[IB2])    / (4.0 * DELTA_X)
    )


def scheme_ftcs_3th(u: np.ndarray) -> np.ndarray:
    """
    Esquema misto 3ª ordem (Solve_Estabilidade_FTCS_3thCS):
        u(i,n+1) = u(i,n) - C·ΔT·[
            (u(i+2) - 2·u(i) + u(i-2))/(4·ΔX)
          + (-4·u(i-1) + 5·u(i) - u(i+2))/(3·ΔX)
        ]
    """
    return u[IC] - (C * DELTA_T) * (
        (u[IF2] - 2*u[IC_4] + u[IB2])         / (4.0 * DELTA_X) +
        (-4*u[IB_4] + 5*u[IC_4] - u[IF2])     / (3.0 * DELTA_X)
    )


def scheme_ftcs_4th(u: np.ndarray) -> np.ndarray:
    """
    Diferença 4ª ordem com parâmetro mi=0.7 (Solve_Estabilidade_FTCS_4thCS):
        u(i,n+1) = u(i,n) - (mi/12)·[-u(i+2) + 8·u(i+1) - 8·u(i-1) + u(i-2)]
    """
    mi = 0.7
    return u[IC] - (mi / 12.0) * (
        -u[IF2] + 8.0*u[IF_4] - 8.0*u[IB_4] + u[IB2]
    )


def scheme_ctcs_2th(u: np.ndarray, um: np.ndarray) -> np.ndarray:
    """
    CTCS — Leap-frog 2ª ordem (Solve_Estabilidade_CTCS_2thCS).
    Fórmula do Fortran (preservada com bug original):
        termX(xc) = u(xb) - C·ΔT/(2·ΔX)·[u(xf) - u(xb)]
    Bug: usa u(xb) em vez de u(xc) no lado esquerdo.
    Leap-frog correto seria: um(xc) - C·ΔT/ΔX·[u(xf) - u(xb)]
    """
    return u[IB] - (C * DELTA_T / (2.0 * DELTA_X)) * (u[IF_] - u[IB])


def scheme_estavel_4th(u: np.ndarray) -> np.ndarray:
    """
    Expansão de Taylor 4ª ordem (Solve_Estavel_4thCS) — ESQUEMA ATIVO.

    Derivadas espaciais (fórmulas ativas no Fortran, não as comentadas):
        du/dx   = [u(i) - u(i-1)] / ΔX                              (backward 1ª ordem)
        d²u/dx² = [u(i+1) - 2u(i) + u(i-1)] / (2·ΔX²)              (centrado 2ª ordem)
        d⁴u/dx⁴ = [u(i+2)-4u(i+1)+6u(i)-4u(i-1)+u(i-2)] / ΔX⁴     (centrado 4ª ordem)

    BUG identificado no Fortran para d³u/dx³:
        Fortran:  [(u(i+2)-u(i-2)) - 2·(u(i+1)+u(i-1))] / (2·ΔX³)
                = [+u(i+2) - 2u(i+1) - 2u(i-1) - u(i-2)] / (2·ΔX³)  ← u(i-1) com sinal errado
        Correto:   [u(i+2) - 2u(i+1) + 2u(i-1) - u(i-2)] / (2·ΔX³)  ← diferença centrada padrão
        Verificação: u=x² → Fortran dá −32, correto dá 0 ✓
                     u=x³ → Fortran dá −122, correto dá 6 ✓

    Esquema (série de Taylor truncada na 4ª derivada):
        u(n+1) = u(n) - C·ΔT·(du/dx) + (C²·ΔT²/2)·(d²u/dx²)
                      - (C³·ΔT³/6)·(d³u/dx³) + (C⁴·ΔT⁴/24)·(d⁴u/dx⁴)
    """
    dx  = DELTA_X
    dt  = DELTA_T

    dudx   = (u[IC_4] - u[IB_4]) / dx
    du2dx2 = (u[IF_4] - 2*u[IC_4] + u[IB_4]) / (2.0 * dx**2)
    # BUG no Fortran: ((u(xf2)-u(xb2)) - 2*(u(xf)+u(xb))) dá sinal errado em u(xb)
    # Correto: [u(i+2) - 2u(i+1) + 2u(i-1) - u(i-2)] / (2·dx³)
    du3dx3 = (u[IF2] - 2*u[IF_4] + 2*u[IB_4] - u[IB2]) / (2.0 * dx**3)
    du4dx4 = (u[IF2] - 4*u[IF_4] + 6*u[IC_4] - 4*u[IB_4] + u[IB2]) / dx**4

    return (u[IC_4]
            - C * dt * dudx
            + (C**2 * dt**2 / 2.0)  * du2dx2
            - (C**3 * dt**3 / 6.0)  * du3dx3
            + (C**4 * dt**4 / 24.0) * du4dx4)


# ===========================================================================
# Loop principal — equivalente à SUBROUTINE Run
# ===========================================================================

def run(scheme_name: str = "estavel_4th") -> dict:
    """
    Executa a simulação para um dado esquema numérico.

    Parâmetros
    ----------
    scheme_name : um dos nomes abaixo:
        "upwind"      → FTBS 1ª ordem
        "ftcs_2th"    → FTCS centrado 2ª ordem
        "ftcs_2th2"   → FTCS misto experimental
        "ftcs_3th"    → FTCS misto 3ª ordem
        "ftcs_4th"    → FTCS 4ª ordem com mi
        "ctcs_2th"    → Leap-frog (com bug original)
        "estavel_4th" → Taylor 4ª ordem  ← padrão (ativo no Fortran)
    """
    state = init_fields()
    u     = state["u"].copy()
    ua    = state["ua"].copy()
    um    = u.copy()

    snap_iters = [0, 50, 100, 200, NITER]
    snaps_u    = {}
    snaps_ua   = {}

    # Salva instante inicial
    snaps_u[0]  = u.copy()
    snaps_ua[0] = ua.copy()

    err_total = 0.0

    for it in range(1, NITER + 1):

        # Solução analítica
        termXa = analytic_function(state, it)

        # Esquema numérico
        if scheme_name == "upwind":
            termX = scheme_upwind_1th(u)
        elif scheme_name == "ftcs_2th":
            termX = scheme_ftcs_2th(u)
        elif scheme_name == "ftcs_2th2":
            termX = scheme_ftcs_2th2(u)
        elif scheme_name == "ftcs_3th":
            termX = scheme_ftcs_3th(u)
        elif scheme_name == "ftcs_4th":
            termX = scheme_ftcs_4th(u)
        elif scheme_name == "ctcs_2th":
            termX = scheme_ctcs_2th(u, um)
        else:  # estavel_4th (padrão)
            termX = scheme_estavel_4th(u)

        # Atualiza campos (equivalente ao DO i=1,Idim no Fortran)
        um[:] = u
        u[:]  = termX
        ua[:] = termXa

        # Acumula erro quadrático
        err_total += np.sum((u - ua)**2)

        if it in snap_iters:
            snaps_u[it]  = u.copy()
            snaps_ua[it] = ua.copy()

    err_mean = err_total / NITER
    print(f"  err={err_mean:.6e}  DeltaX={DELTA_X}  DeltaT={DELTA_T}  "
          f"CFL={CFL:.4f}  esquema={scheme_name}")

    return {
        "scheme"   : scheme_name,
        "xa"       : state["xa"],
        "snaps_u"  : snaps_u,
        "snaps_ua" : snaps_ua,
        "err_mean" : err_mean,
    }


# ===========================================================================
# Visualizações didáticas
# ===========================================================================

SCHEME_LABELS = {
    "upwind"      : "FTBS UpWind 1ª ordem",
    "ftcs_2th"    : "FTCS Centrado 2ª ordem",
    "ftcs_2th2"   : "FTCS Misto Experimental",
    "ftcs_3th"    : "FTCS Misto 3ª ordem",
    "ftcs_4th"    : "FTCS 4ª ordem (mi=0.7)",
    "ctcs_2th"    : "CTCS Leap-Frog 2ª ordem",
    "estavel_4th" : "Taylor 4ª ordem (ativo)",
}

SCHEME_COLORS = {
    "upwind"      : "#e74c3c",
    "ftcs_2th"    : "#3498db",
    "ftcs_2th2"   : "#9b59b6",
    "ftcs_3th"    : "#1abc9c",
    "ftcs_4th"    : "#e67e22",
    "ctcs_2th"    : "#95a5a6",
    "estavel_4th" : "#27ae60",
}


def plot_single_scheme(sim: dict):
    """
    Quatro painéis para o esquema ativo (Taylor 4ª ordem):
      1. Evolução temporal (snapshots)
      2. Comparação no passo final: numérico vs analítico
      3. Erro pontual ao longo do domínio (passo final)
      4. Integral do erro quadrático acumulado estimada
    """
    xa        = sim["xa"]
    snaps_u   = sim["snaps_u"]
    snaps_ua  = sim["snaps_ua"]
    scheme    = sim["scheme"]
    label     = SCHEME_LABELS[scheme]
    color     = SCHEME_COLORS[scheme]
    C_ANA     = "#2c3e50"

    snap_keys = sorted(snaps_u.keys())

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#f7f9fc")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ----- Painel 1: evolução temporal -----
    ax1 = fig.add_subplot(gs[0, 0])
    alphas = np.linspace(0.2, 1.0, len(snap_keys))
    for alpha, it in zip(alphas, snap_keys):
        ax1.plot(xa, snaps_u[it], color=color, alpha=alpha,
                 lw=1.8, label=f"it={it}")
    ax1.set_title(f"Evolução Temporal\n{label}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("x", fontsize=10); ax1.set_ylabel("u(x, t)", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#fdfdfd")

    # ----- Painel 2: numérico vs analítico (passo final) -----
    ax2 = fig.add_subplot(gs[0, 1])
    it_f = snap_keys[-1]
    ax2.plot(xa, snaps_ua[it_f], color=C_ANA, lw=2.5, label="Analítica", zorder=4)
    ax2.plot(xa, snaps_u[it_f],  color=color,  lw=2.0, ls="--",
             label=f"Numérico (it={it_f})", zorder=3)
    ax2.set_title(f"Numérico vs Analítico  (it={it_f})\n{label}",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("x", fontsize=10); ax2.set_ylabel("u(x, t)", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#fdfdfd")

    # ----- Painel 3: erro pontual -----
    ax3 = fig.add_subplot(gs[1, 0])
    erro_pont = snaps_u[it_f] - snaps_ua[it_f]
    ax3.fill_between(xa, erro_pont, alpha=0.35, color=color)
    ax3.plot(xa, erro_pont, color=color, lw=1.8)
    ax3.axhline(0, color="gray", lw=0.8, ls=":")
    rms = np.sqrt(np.mean(erro_pont**2))
    ax3.set_title(f"Erro Pontual  u_num − u_ana  (it={it_f})\nRMSE = {rms:.4e}",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("x", fontsize=10); ax3.set_ylabel("Erro", fontsize=10)
    ax3.grid(True, alpha=0.3); ax3.set_facecolor("#fdfdfd")

    # ----- Painel 4: condição inicial vs. analítica no passo final -----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(xa, snaps_u[0],    color="#7f8c8d", lw=2.0, ls=":",
             label="Condição inicial", zorder=2)
    ax4.plot(xa, snaps_ua[it_f], color=C_ANA,  lw=2.5,
             label=f"Analítica it={it_f}", zorder=4)
    ax4.plot(xa, snaps_u[it_f],  color=color,  lw=2.0, ls="--",
             label=f"Numérico it={it_f}", zorder=3)
    ax4.set_title("Deformação do Pulso\n(Inicial → Analítica → Numérica)",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel("x", fontsize=10); ax4.set_ylabel("u(x, t)", fontsize=10)
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)
    ax4.set_facecolor("#fdfdfd")
    ax4.text(0.02, 0.97,
             f"CFL = {CFL:.2f}\nΔX = {DELTA_X}  ΔT = {DELTA_T}\nErr médio = {sim['err_mean']:.4e}",
             transform=ax4.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#95a5a6", alpha=0.9))

    fig.suptitle(
        f"Advecção Linear 1D — Pulso Top-Hat\n"
        r"$\partial u/\partial t + C\,\partial u/\partial x = 0$"
        f"    C={C} m/s,  CFL={CFL:.2f},  {NITER} iterações",
        fontsize=13, fontweight="bold", y=0.99
    )
    out = f"/mnt/user-data/outputs/advec_tophat_{scheme}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Figura salva: advec_tophat_{scheme}.png")
    plt.show()


def plot_comparison(sims: dict):
    """
    Comparação de todos os esquemas estáveis no passo final.
    """
    C_ANA = "#2c3e50"
    xa    = next(iter(sims.values()))["xa"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.patch.set_facecolor("#f7f9fc")
    axes = axes.flatten()

    schemes_plot = [k for k in sims if k != "ftcs_2th2"]  # remove instável experimental
    it_f = sorted(next(iter(sims.values()))["snaps_u"].keys())[-1]

    for ax, sch in zip(axes, schemes_plot):
        sim   = sims[sch]
        color = SCHEME_COLORS[sch]
        label = SCHEME_LABELS[sch]

        ax.plot(xa, sim["snaps_ua"][it_f], color=C_ANA, lw=2.5,
                label="Analítica", zorder=4)
        ax.plot(xa, sim["snaps_u"][it_f],  color=color,  lw=2.0, ls="--",
                label="Numérico", zorder=3)

        rms = np.sqrt(np.mean((sim["snaps_u"][it_f] - sim["snaps_ua"][it_f])**2))
        ax.set_title(f"{label}\nRMSE={rms:.3e}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("x", fontsize=9); ax.set_ylabel("u", fontsize=9)
        ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)
        ax.set_facecolor("#fdfdfd")

    # Eixo extra: tabela de erros
    axes[-1].axis("off")
    rows = [[SCHEME_LABELS[k],
             f"{sims[k]['err_mean']:.4e}",
             "✓" if k == "estavel_4th" else ""]
            for k in schemes_plot if k in sims]
    tbl = axes[-1].table(
        cellText=rows,
        colLabels=["Esquema", "Erro Médio", "Ativo"],
        cellLoc="center", loc="center",
        bbox=[0, 0.1, 1, 0.85],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for j in range(3):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        tbl[(i, 0)].set_facecolor("#ecf0f1")
    axes[-1].set_title("Resumo de Erros\n(it final)", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Comparação de Esquemas — Advecção 1D Pulso Top-Hat\n"
        f"CFL={CFL:.2f},  it={it_f}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = "/mnt/user-data/outputs/advec_tophat_comparacao.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Figura salva: advec_tophat_comparacao.png")
    plt.show()


# ===========================================================================
# Execução principal
# ===========================================================================
if __name__ == "__main__":

    print("=" * 62)
    print("  Advecção Linear 1D — Pulso Top-Hat")
    print("=" * 62)
    print(f"  Idim   = {IDIM}")
    print(f"  DeltaX = {DELTA_X}")
    print(f"  DeltaT = {DELTA_T}")
    print(f"  C      = {C} m/s")
    print(f"  CFL    = {CFL:.4f}")
    print(f"  niter  = {NITER}")
    print("=" * 62)

    # Esquemas a simular (omite ctcs_2th por ter bug severo de divergência)
    schemes_to_run = [
        "estavel_4th",   # ← ativo no Fortran
        "upwind",
        "ftcs_2th",
        "ftcs_3th",
        "ftcs_4th",
        "ctcs_2th",
    ]

    sims = {}
    for sch in schemes_to_run:
        print(f"\n  [{sch}]")
        try:
            sims[sch] = run(sch)
        except Exception as e:
            print(f"    ERRO: {e}")

    print("\n  Gerando figura do esquema ativo (Taylor 4ª ordem)...")
    plot_single_scheme(sims["estavel_4th"])

    print("\n  Gerando figura comparativa de todos os esquemas...")
    plot_comparison(sims)
