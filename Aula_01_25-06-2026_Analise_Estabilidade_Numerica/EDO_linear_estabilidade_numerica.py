"""
=============================================================================
Aula 01 - 25/06/2026
Análise de Estabilidade Numérica — Esquema de Euler Explícito (Forward)
=============================================================================
Conversão do módulo Fortran Class_NumericalScheme para Python.

EDO de decaimento exponencial:
    dy/dt = -lambda * y,   y(0) = y0

Solução analítica:
    y(t) = y0 * exp(-lambda * t)

Esquema numérico (Euler Forward / Diferença Progressiva):
    y(n+1) = y(n) - lambda * dt * y(n)
           = y(n) * (1 - lambda * dt)

Fator de amplificação: A = 1 - lambda * dt
Condição de estabilidade: |A| <= 1  =>  0 < lambda * dt <= 2
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Parâmetros globais (equivalentes às PARAMETERs do Fortran)
# ---------------------------------------------------------------------------
LAMBDA = 0.1      # [1/s]  taxa de decaimento
Y0     = 1.0      # condição inicial
NREC   = 200      # número de passos de tempo


# ===========================================================================
# Classe NumericalScheme  (equivalente ao MODULE Class_NumericalScheme)
# ===========================================================================
class NumericalScheme:
    """
    Encapsula o esquema de Euler Forward para a EDO de decaimento:
        dy/dt = -lambda * y
    """

    def __init__(self, dt: float, lam: float):
        """
        Inicializa o esquema numérico.

        Parâmetros
        ----------
        dt  : passo de tempo [s]
        lam : coeficiente de decaimento lambda [1/s]
        """
        self.dt  = dt
        self.lam = lam

    def scheme_forward(self, yc: float) -> float:
        """
        Esquema de Euler Explícito (Forward):

            y(n+1) - y(n)
            ------------- = -lambda * y(n)
                  dt

        Retorna y(n+1).
        """
        return yc - self.lam * self.dt * yc   # = yc * (1 - lambda*dt)

    @staticmethod
    def scheme_update(y_in: float) -> float:
        """
        Atualiza o valor atual (pass-through, mantém lógica original).
        """
        return y_in

    def analytic_function(self, y0: float, tn: int) -> float:
        """
        Solução analítica da EDO:
            y(t) = y0 * exp(-lambda * tn * dt)
        """
        return y0 * np.exp(-self.lam * tn * self.dt)

    @property
    def amplification_factor(self) -> float:
        """
        Fator de amplificação do esquema:  A = 1 - lambda * dt
        Estabilidade exige |A| <= 1.
        """
        return 1.0 - self.lam * self.dt


# ===========================================================================
# Programa principal  (equivalente ao PROGRAM MAIN)
# ===========================================================================
def run_simulation(dt: float) -> dict:
    """
    Executa a simulação para um dado passo de tempo dt.

    Retorna dicionário com arrays de tempo, solução numérica e analítica.
    """
    scheme = NumericalScheme(dt=dt, lam=LAMBDA)

    yc      = Y0
    t_vals  = []
    yn_vals = []
    ya_vals = []

    for tn in range(NREC + 1):
        yp = scheme.scheme_forward(yc)
        ya = scheme.analytic_function(Y0, tn)

        t_vals.append(tn * dt)
        yn_vals.append(yc)
        ya_vals.append(ya)

        # yn = scheme_update(yc)   → mantido mas não altera o fluxo
        yc = scheme.scheme_update(yp)

    return {
        "dt"    : dt,
        "A"     : scheme.amplification_factor,
        "t"     : np.array(t_vals),
        "yn"    : np.array(yn_vals),
        "ya"    : np.array(ya_vals),
        "scheme": scheme,
    }


# ===========================================================================
# Visualizações didáticas
# ===========================================================================
def plot_results():
    """
    Gera quatro painéis didáticos:
      1. Solução numérica vs analítica (dt original do Fortran)
      2. Erro absoluto ao longo do tempo
      3. Comparação entre dt estável, crítico e instável
      4. Diagrama de estabilidade: fator de amplificação vs lambda*dt
    """

    # --- Três cenários de dt ------------------------------------------------
    # Fortran original: dt = 0.5 / lambda = 5 s  →  A = 1 - 0.1*5 = 0.5  (estável)
    dt_stable   = 0.5 / LAMBDA          # dt = 5 s,  A = 0.50  (estável)
    dt_critical = 2.0 / LAMBDA          # dt = 20 s, A = -1.0  (limite)
    dt_unstable = 2.5 / LAMBDA          # dt = 25 s, A = -1.5  (instável)

    sim_stable   = run_simulation(dt_stable)
    sim_critical = run_simulation(dt_critical)
    sim_unstable = run_simulation(dt_unstable)

    # --- Configuração de figura ---------------------------------------------
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("#f7f9fc")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    colors = {
        "analitica" : "#2c3e50",
        "estavel"   : "#27ae60",
        "critico"   : "#e67e22",
        "instavel"  : "#e74c3c",
        "erro"      : "#8e44ad",
    }

    # =========================================================
    # Painel 1 — Solução numérica vs analítica (caso estável)
    # =========================================================
    ax1 = fig.add_subplot(gs[0, 0])
    s = sim_stable
    ax1.plot(s["t"], s["ya"], color=colors["analitica"], lw=2.5,
             label=r"$y_a(t) = e^{-\lambda t}$  (analítica)", zorder=3)
    ax1.plot(s["t"], s["yn"], color=colors["estavel"], lw=1.8,
             ls="--", marker="o", markersize=2,
             label=fr"Euler Forward  ($\Delta t={s['dt']:.1f}$ s, $A={s['A']:.2f}$)")
    ax1.set_title("Solução Numérica vs Analítica\n(caso estável)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Tempo [s]", fontsize=10)
    ax1.set_ylabel("y(t)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, s["t"][-1])
    ax1.grid(True, alpha=0.35)
    ax1.set_facecolor("#fdfdfd")

    # Anotação: critério de estabilidade
    ldt = LAMBDA * s["dt"]
    ax1.text(0.97, 0.95,
             fr"$\lambda \Delta t = {ldt:.2f}$" + "\n" + fr"$A = 1 - \lambda\Delta t = {s['A']:.2f}$",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=colors["estavel"], alpha=0.85))

    # =========================================================
    # Painel 2 — Erro absoluto ao longo do tempo
    # =========================================================
    ax2 = fig.add_subplot(gs[0, 1])
    erro = np.abs(s["yn"] - s["ya"])
    ax2.semilogy(s["t"], np.where(erro == 0, np.nan, erro),
                 color=colors["erro"], lw=2, marker="o", markersize=2.5)
    ax2.set_title("Erro Absoluto  |$y_n - y_a$|\n(escala logarítmica)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tempo [s]", fontsize=10)
    ax2.set_ylabel(r"$|y_n - y_a|$", fontsize=10)
    ax2.grid(True, which="both", alpha=0.35)
    ax2.set_facecolor("#fdfdfd")
    ax2.set_xlim(0, s["t"][-1])

    # Anotação: acúmulo de erro
    ax2.text(0.05, 0.92,
             "Erro cresce com o tempo\n(truncamento + propagação)",
             transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=colors["erro"], alpha=0.85))

    # =========================================================
    # Painel 3 — Comparação estável / crítico / instável
    # =========================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # analítica de referência (resolução fina)
    t_fine = np.linspace(0, sim_stable["t"][-1], 800)
    ax3.plot(t_fine, Y0 * np.exp(-LAMBDA * t_fine),
             color=colors["analitica"], lw=2.5, label="Analítica", zorder=4)

    for sim, cor, lbl in [
        (sim_stable,   colors["estavel"],  fr"Estável   $\Delta t={dt_stable:.0f}$ s  (A={sim_stable['A']:.2f})"),
        (sim_critical, colors["critico"],  fr"Crítico   $\Delta t={dt_critical:.0f}$ s  (A={sim_critical['A']:.2f})"),
        (sim_unstable, colors["instavel"], fr"Instável  $\Delta t={dt_unstable:.0f}$ s  (A={sim_unstable['A']:.2f})"),
    ]:
        mask = sim["t"] <= sim_stable["t"][-1]
        ax3.plot(sim["t"][mask], sim["yn"][mask], color=cor, lw=1.8,
                 ls="--", marker="o", markersize=2.5, label=lbl, zorder=3)

    ax3.set_title("Comparação de Estabilidade\n(três regimes de $\\Delta t$)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Tempo [s]", fontsize=10)
    ax3.set_ylabel("y(t)", fontsize=10)
    ax3.legend(fontsize=8.5, loc="upper right")
    ax3.set_ylim(-2.5, 1.5)
    ax3.set_xlim(0, sim_stable["t"][-1])
    ax3.axhline(0, color="gray", lw=0.8, ls=":")
    ax3.grid(True, alpha=0.35)
    ax3.set_facecolor("#fdfdfd")

    # =========================================================
    # Painel 4 — Diagrama de estabilidade: A vs lambda*dt
    # =========================================================
    ax4 = fig.add_subplot(gs[1, 1])

    ldt_vals = np.linspace(0, 3.0, 400)
    A_vals   = 1.0 - ldt_vals

    # Regiões de estabilidade
    stable_mask   = np.abs(A_vals) <= 1
    unstable_mask = ~stable_mask

    ax4.axhspan(-1, 1, alpha=0.12, color=colors["estavel"], label="Região estável  |A| ≤ 1")
    ax4.axhspan( 1, 2, alpha=0.10, color=colors["instavel"])
    ax4.axhspan(-2,-1, alpha=0.10, color=colors["instavel"], label="Região instável |A| > 1")

    ax4.plot(ldt_vals, A_vals, color="#2c3e50", lw=2.5, label=r"$A = 1 - \lambda\Delta t$")
    ax4.axhline( 1, color="gray", lw=1, ls=":")
    ax4.axhline(-1, color="gray", lw=1, ls=":")
    ax4.axhline( 0, color="gray", lw=0.6, ls=":")
    ax4.axvline( 2, color=colors["instavel"], lw=1.5, ls="--", label=r"$\lambda\Delta t = 2$ (limite)")

    # Marcadores dos três cenários
    for sim, cor, mk in [
        (sim_stable,   colors["estavel"],  "o"),
        (sim_critical, colors["critico"],  "s"),
        (sim_unstable, colors["instavel"], "^"),
    ]:
        ldt_pt = LAMBDA * sim["dt"]
        A_pt   = sim["A"]
        ax4.scatter(ldt_pt, A_pt, color=cor, s=90, zorder=5, marker=mk,
                    edgecolors="white", linewidths=0.8)
        ax4.annotate(fr"$\lambda\Delta t={ldt_pt:.1f}$, A={A_pt:.1f}",
                     xy=(ldt_pt, A_pt), xytext=(ldt_pt + 0.08, A_pt + 0.12),
                     fontsize=8, color=cor,
                     arrowprops=dict(arrowstyle="->", color=cor, lw=0.8))

    ax4.set_title("Diagrama de Estabilidade\n(Euler Forward)", fontsize=12, fontweight="bold")
    ax4.set_xlabel(r"$\lambda \, \Delta t$", fontsize=11)
    ax4.set_ylabel(r"Fator de amplificação  $A$", fontsize=10)
    ax4.legend(fontsize=8.5, loc="upper right")
    ax4.set_xlim(0, 3.0)
    ax4.set_ylim(-2.2, 1.5)
    ax4.grid(True, alpha=0.30)
    ax4.set_facecolor("#fdfdfd")

    # =========================================================
    # Título geral
    # =========================================================
    fig.suptitle(
        "Aula 01 — Análise de Estabilidade Numérica\n"
        r"EDO de Decaimento:  $\dfrac{dy}{dt} = -\lambda\,y$  "
        r"— Esquema de Euler Forward",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.savefig("/mnt/user-data/outputs/edo_estabilidade_numerica.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Figura salva: edo_estabilidade_numerica.png")
    plt.show()


# ===========================================================================
# Execução principal
# ===========================================================================
if __name__ == "__main__":

    # Configuração idêntica ao MAIN do Fortran
    dt_fortran = 0.5 / LAMBDA          # = 5.0 s
    scheme = NumericalScheme(dt=dt_fortran, lam=LAMBDA)

    print("=" * 60)
    print("  Análise de Estabilidade — Euler Forward")
    print("=" * 60)
    print(f"  lambda          = {LAMBDA}  [1/s]")
    print(f"  dt (Fortran)    = {dt_fortran:.2f}  [s]")
    print(f"  lambda * dt     = {LAMBDA * dt_fortran:.4f}")
    print(f"  Fator A         = {scheme.amplification_factor:.4f}")
    print(f"  Condição |A|<=1 : {'✓ ESTÁVEL' if abs(scheme.amplification_factor) <= 1 else '✗ INSTÁVEL'}")
    print(f"  Número de passos: {NREC}")
    print("=" * 60)

    sim = run_simulation(dt_fortran)

    print(f"\n  Primeiros 5 valores:")
    print(f"  {'tn':>4}  {'t [s]':>8}  {'y_num':>12}  {'y_ana':>12}  {'erro abs':>12}")
    print("  " + "-" * 54)
    for i in range(5):
        err = abs(sim["yn"][i] - sim["ya"][i])
        print(f"  {i:>4}  {sim['t'][i]:>8.2f}  {sim['yn'][i]:>12.6f}  "
              f"{sim['ya'][i]:>12.6f}  {err:>12.2e}")

    print("\n  Gerando figuras diagnósticas...")
    plot_results()
