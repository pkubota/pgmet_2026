"""
=============================================================================
Aula 01 — 25/06/2026
Fontes de Erros Numéricos — Precisão de Ponto Flutuante
=============================================================================
Conversão do programa Fortran PROGRAM Main para Python.

O programa demonstra o efeito do cancelamento catastrófico e da
saturação de representação em ponto flutuante ao acumular N=1e8
somas de 1.0 em precisão simples (float32) e dupla (float64).

Conceito central:
  - float32 tem ~7 dígitos decimais de precisão (mantissa 23 bits)
  - float64 tem ~15 dígitos decimais de precisão (mantissa 52 bits)
  - A partir de certo valor, float32 não consegue representar
    x + 1.0 > x  →  a soma "congela" (stagnation / saturation)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Parâmetros (idênticos ao Fortran)
# ---------------------------------------------------------------------------
N             = int(1e8)
SINGLE_KIND   = np.float32    # KIND=4  →  float32
DOUBLE_KIND   = np.float64    # KIND=8  →  float64

# ===========================================================================
# 1. Simulação completa (equivalente ao DO loop do Fortran)
#    Aqui fazemos duas coisas:
#      a) Detectamos o ponto de saturação do float32
#      b) Calculamos soma total em ambas as precisões
# ===========================================================================

def find_saturation_point() -> int:
    """
    Encontra o passo i em que float32 satura:
        x + 1.0 == x   (a unidade não pode mais ser representada)
    Equivale ao IF comentado no Fortran:
        IF(s_X == s_X_m) print*, s_X, d_X
    """
    x = SINGLE_KIND(0.0)
    for i in range(1, N + 1):
        x_prev = x
        x = SINGLE_KIND(x + SINGLE_KIND(1.0))
        if x == x_prev:
            return i, x
    return N, x


def run_accumulation_loop(n_steps: int, track_every: int = 500_000):
    """
    Executa o loop de acumulação e coleta amostras ao longo do tempo.

    Retorna
    -------
    s_X   : soma final em float32
    d_X   : soma final em float64
    steps : array de índices amostrados
    s_vals: valores float32 ao longo dos passos
    d_vals: valores float64 ao longo dos passos
    """
    s_X  = SINGLE_KIND(0.0)
    d_X  = DOUBLE_KIND(0.0)

    steps  = []
    s_vals = []
    d_vals = []

    for i in range(1, n_steps + 1):
        s_X = SINGLE_KIND(s_X + SINGLE_KIND(1.0))
        d_X = DOUBLE_KIND(d_X + DOUBLE_KIND(1.0))

        if i % track_every == 0:
            steps.append(i)
            s_vals.append(float(s_X))
            d_vals.append(float(d_X))

    return s_X, d_X, np.array(steps), np.array(s_vals), np.array(d_vals)


# ---------------------------------------------------------------------------
# Etapa 1 — Encontrar saturação (loop leve)
# ---------------------------------------------------------------------------
print("=" * 62)
print("  Fonte de Erros — Precisão de Ponto Flutuante")
print("=" * 62)
print(f"  N = {N:,}  passos de acumulação")
print()
print("  [1/3] Detectando ponto de saturação do float32...")
sat_step, sat_val = find_saturation_point()
print(f"        → Saturação em i = {sat_step:,}  (s_X = {sat_val:.1f})")
print(f"        → 2^24 = {2**24:,}  (limite teórico da mantissa float32)")

# ---------------------------------------------------------------------------
# Etapa 2 — Loop completo com amostragem
# ---------------------------------------------------------------------------
print()
print("  [2/3] Executando loop completo (N = 1e8)...")
print("        (pode levar alguns segundos...)")

s_X, d_X, steps, s_vals, d_vals = run_accumulation_loop(N)

# Resultados idênticos ao PRINT* do Fortran
print()
print("  === Resultado equivalente ao Fortran PRINT* ===")
print(f"  soma_s_X  = {float(s_X):.6g}    media_s_X = {float(s_X)/N:.6g}")
print(f"  soma_d_X  = {float(d_X):.6g}  media_d_X = {float(d_X)/N:.6g}")
print()

erro_rel_s = abs(float(s_X) - N) / N
erro_rel_d = abs(float(d_X) - N) / N
print(f"  Valor exato esperado : {N:,}")
print(f"  Erro relativo float32: {erro_rel_s:.4%}")
print(f"  Erro relativo float64: {erro_rel_d:.4%}")
print()

# ===========================================================================
# 3. Visualizações didáticas
# ===========================================================================
print("  [3/3] Gerando figuras diagnósticas...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#f7f9fc")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

C32  = "#e74c3c"    # vermelho — float32
C64  = "#2980b9"    # azul    — float64
CEXATO = "#2c3e50"  # cinza   — valor exato
CSAT   = "#e67e22"  # laranja — saturação

# =========================================================
# Painel 1 — Evolução da soma acumulada
# =========================================================
ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(steps / 1e6, d_vals / 1e6, color=C64,   lw=2.2, label="float64 (double_kind)", zorder=3)
ax1.plot(steps / 1e6, s_vals / 1e6, color=C32,   lw=2.0, ls="--", label="float32 (single_kind)", zorder=3)
ax1.plot([0, N/1e6], [0, N/1e6],    color=CEXATO, lw=1.2, ls=":", label="Valor exato", zorder=2)

ax1.axvline(sat_step / 1e6, color=CSAT, lw=1.5, ls="--",
            label=f"Saturação float32\n(i ≈ {sat_step/1e6:.0f}M = 2²⁴)")

ax1.set_title("Soma Acumulada: float32 vs float64", fontsize=12, fontweight="bold")
ax1.set_xlabel("Passo i  [×10⁶]", fontsize=10)
ax1.set_ylabel("Soma acumulada  [×10⁶]", fontsize=10)
ax1.legend(fontsize=8.5)
ax1.grid(True, alpha=0.35)
ax1.set_facecolor("#fdfdfd")

# =========================================================
# Painel 2 — Erro absoluto acumulado
# =========================================================
ax2 = fig.add_subplot(gs[0, 1])

erro_s = np.abs(s_vals - steps)
erro_d = np.abs(d_vals - steps)
# evitar log(0)
erro_s = np.where(erro_s == 0, np.nan, erro_s)
erro_d = np.where(erro_d == 0, np.nan, erro_d)

ax2.semilogy(steps / 1e6, erro_s, color=C32, lw=2.0, ls="--", label="float32 — erro abs")
ax2.semilogy(steps / 1e6, erro_d, color=C64, lw=2.0,           label="float64 — erro abs")
ax2.axvline(sat_step / 1e6, color=CSAT, lw=1.5, ls="--",
            label=f"Saturação float32")

ax2.set_title("Erro Absoluto  |soma − i|\n(escala logarítmica)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Passo i  [×10⁶]", fontsize=10)
ax2.set_ylabel("|soma − i|", fontsize=10)
ax2.legend(fontsize=8.5)
ax2.grid(True, which="both", alpha=0.35)
ax2.set_facecolor("#fdfdfd")

# =========================================================
# Painel 3 — Zoom na região de saturação float32
# =========================================================
ax3 = fig.add_subplot(gs[1, 0])

# Região de interesse: entorno do ponto 2^24
zoom_start = max(0, sat_step - int(5e6))
zoom_end   = min(N, sat_step + int(30e6))

mask = (steps >= zoom_start) & (steps <= zoom_end)
ax3.plot(steps[mask] / 1e6, d_vals[mask] / 1e6, color=C64, lw=2.0, label="float64")
ax3.plot(steps[mask] / 1e6, s_vals[mask] / 1e6, color=C32, lw=2.0, ls="--", label="float32 (congela)")
ax3.plot([zoom_start/1e6, zoom_end/1e6],
         [zoom_start/1e6, zoom_end/1e6], color=CEXATO, lw=1.2, ls=":", label="Exato")

ax3.axvline(sat_step / 1e6, color=CSAT, lw=1.5, ls="--", label=f"Saturação (i={sat_step/1e6:.1f}M)")

# Anotação de "congelamento"
ax3.annotate("float32 congela\n(soma não cresce mais)",
             xy=(sat_step/1e6, float(sat_val)/1e6),
             xytext=(sat_step/1e6 + 5, float(sat_val)/1e6 - 8),
             fontsize=8.5, color=CSAT,
             arrowprops=dict(arrowstyle="->", color=CSAT, lw=1.0))

ax3.set_title("Zoom — Região de Saturação\n(perto de $2^{24}$ ≈ 16.7M)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Passo i  [×10⁶]", fontsize=10)
ax3.set_ylabel("Soma acumulada  [×10⁶]", fontsize=10)
ax3.legend(fontsize=8.5)
ax3.grid(True, alpha=0.35)
ax3.set_facecolor("#fdfdfd")

# =========================================================
# Painel 4 — Tabela comparativa de resultados
# =========================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")

# Máquina epsilon (ULP — unidade de menor precisão)
eps32 = np.finfo(np.float32).eps
eps64 = np.finfo(np.float64).eps

tabela = [
    ["Parâmetro",               "float32 (KIND=4)",     "float64 (KIND=8)"],
    ["Bits de mantissa",        "23 bits",              "52 bits"],
    ["Dígitos decimais (≈)",    "~7",                   "~15"],
    ["Machine epsilon (ε)",     f"{eps32:.2e}",         f"{eps64:.2e}"],
    ["Limite de saturação",     f"2²⁴ ≈ {2**24/1e6:.1f}M",  "—"],
    [f"Soma após N={N//int(1e6)}M",  f"{float(s_X):.6g}",  f"{float(d_X):.6g}"],
    ["Valor exato esperado",    f"{N:,}",               f"{N:,}"],
    ["Erro relativo",           f"{erro_rel_s:.4%}",    f"{erro_rel_d:.4%}"],
    ["Média (soma/N)",          f"{float(s_X)/N:.6g}",  f"{float(d_X)/N:.6g}"],
]

tbl = ax4.table(
    cellText=tabela[1:],
    colLabels=tabela[0],
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)

# Estilo do cabeçalho
for j in range(3):
    tbl[(0, j)].set_facecolor("#2c3e50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# Colorir colunas
for i in range(1, len(tabela)):
    tbl[(i, 0)].set_facecolor("#ecf0f1")
    tbl[(i, 0)].set_text_props(fontweight="bold")
    tbl[(i, 1)].set_facecolor("#fdecea")   # vermelho suave — float32
    tbl[(i, 2)].set_facecolor("#eaf4fb")   # azul suave    — float64

ax4.set_title("Comparativo de Precisão Numérica", fontsize=12, fontweight="bold", pad=12)

# =========================================================
# Título geral
# =========================================================
fig.suptitle(
    "Aula 01 — Fontes de Erros Numéricos\n"
    r"Saturação em Ponto Flutuante: acumulação de $\sum_{i=1}^{N} 1.0$  "
    r"com $N = 10^8$",
    fontsize=13, fontweight="bold", y=0.99
)

out_path = "/mnt/user-data/outputs/precisao_ponto_flutuante.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  Figura salva: precisao_ponto_flutuante.png")
plt.show()
