"""
Cálculo e visualização das medidas estatísticas de uma distribuição:
posição (média, mediana, moda), dispersão (variância, desvio padrão)
e forma (assimetria, curtose).

Usa o mesmo conjunto de dados do exemplo passo a passo:
X = {2, 4, 4, 4, 5, 5, 7, 9}
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Dados e medidas básicas
# ---------------------------------------------------------------
X = np.array([2, 4, 4, 4, 5, 5, 7, 9])
n = len(X)

media = np.mean(X)
mediana = np.median(X)
moda = stats.mode(X, keepdims=True).mode[0]

variancia = np.mean((X - media) ** 2)          # populacional (divide por n)
desvio_padrao = np.sqrt(variancia)
cv = desvio_padrao / media                      # coeficiente de variação

# Assimetria e curtose (definição de momentos, igual ao passo a passo)
assimetria = np.mean((X - media) ** 3) / desvio_padrao ** 3
curtose_excesso = np.mean((X - media) ** 4) / desvio_padrao ** 4 - 3

print("=== Medidas de posição ===")
print(f"Média:   {media:.3f}")
print(f"Mediana: {mediana:.3f}")
print(f"Moda:    {moda}")
print()
print("=== Medidas de dispersão ===")
print(f"Variância:      {variancia:.3f}")
print(f"Desvio padrão:  {desvio_padrao:.3f}")
print(f"Coef. variação: {cv:.3f}")
print()
print("=== Medidas de forma ===")
print(f"Assimetria (γ1): {assimetria:.3f}")
print(f"Curtose (γ2):    {curtose_excesso:.3f}")

# Conferência usando scipy (deve bater com o cálculo manual)
print()
print("=== Conferência com scipy.stats ===")
print(f"skew (scipy):     {stats.skew(X):.3f}")
print(f"kurtosis (scipy): {stats.kurtosis(X):.3f}")  # já é o excesso (Fisher)

# ---------------------------------------------------------------
# 2. Gráficos
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- (a) Posição: histograma + média/mediana/moda ---
ax = axes[0]
ax.hist(X, bins=np.arange(1.5, 10.5, 1), color="#7FA6D8", edgecolor="white")
ax.axvline(media, color="#D85A30", lw=2, label=f"Média = {media:.1f}")
ax.axvline(mediana, color="#0F6E56", lw=2, ls="--", label=f"Mediana = {mediana:.1f}")
ax.axvline(moda, color="#854F0B", lw=2, ls=":", label=f"Moda = {moda}")
ax.set_title("Medidas de posição")
ax.set_xlabel("x")
ax.set_ylabel("frequência")
ax.legend(fontsize=9)

# --- (b) Dispersão: comparação de duas variâncias ---
ax = axes[1]
x_axis = np.linspace(-6, 6, 400)
for sigma, cor, rotulo in [(1, "#378ADD", "σ = 1"), (2, "#D85A30", "σ = 2")]:
    ax.plot(x_axis, stats.norm.pdf(x_axis, 0, sigma), color=cor, lw=2, label=rotulo)
ax.set_title("Medidas de dispersão\n(variância = largura da curva)")
ax.set_xlabel("x")
ax.set_ylabel("densidade")
ax.legend(fontsize=9)

# --- (c) Forma: assimetria e curtose no mesmo painel ---
ax = axes[2]
grid = np.linspace(-5, 8, 400)
# normal de referência
ax.plot(grid, stats.norm.pdf(grid, 0, 1.3), color="gray", lw=1.5, ls="--", label="normal (ref.)")
# assimétrica positiva (skew-normal)
ax.plot(grid, stats.skewnorm.pdf(grid, a=4, loc=-1.2, scale=1.6),
        color="#534AB7", lw=2, label="assimetria > 0")
# leptocúrtica (t-Student com poucos graus de liberdade)
ax.plot(grid, stats.t.pdf(grid, df=2.2, scale=1.0), color="#D85A30", lw=2, label="curtose > 0")
ax.set_title("Medidas de forma")
ax.set_xlabel("x")
ax.set_ylabel("densidade")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("medidas_estatisticas.png", dpi=150)
print("\nGráfico salvo em medidas_estatisticas.png")
