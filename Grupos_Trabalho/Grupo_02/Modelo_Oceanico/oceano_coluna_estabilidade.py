# -*- coding: utf-8 -*-
"""
MET-576-4 - Estabilidade e dissipacao em esquemas de integracao temporal
Analise de estabilidade linear: Leapfrog vs. Adams-Bashforth-2 vs. RK4

Sistema teste (oscilador de empuxo, reduzido a EDO complexa):
    dz/dt = -i N z,   z = N w + i b

x = N * dt e' o parametro de estabilidade adimensional.

Para cada esquema, deriva-se a relacao de recorrencia linear e suas
raizes caracteristicas lambda(x):

  RK4 (1 passo, 1 raiz):
      w_(n+1) = w_n [1 + z + z^2/2 + z^3/6 + z^4/24],  z = -i x
      (reproduz os 4 primeiros termos de exp(-i x))

  Leapfrog (3 niveis, 2 raizes):
      lambda^2 + 2 i x lambda - 1 = 0
      lambda_fis  = -i x + sqrt(1 - x^2)   (aprox. exp(-ix), fisico)
      lambda_comp = -i x - sqrt(1 - x^2)   (espurio, |lambda_comp|=1
                                             sempre -- NAO amortece)

  Adams-Bashforth-2 (2 niveis, 2 raizes):
      y_(n+1) = y_n + dt/2 (3 f_n - f_(n-1)),  f_n = -i N y_n
      => lambda^2 - (1 + 3z/2) lambda + z/2 = 0,   z = -i x
      lambda_fis  ~ exp(-ix) para x pequeno (raiz principal)
      lambda_comp -> 0 quando x -> 0 (raiz espuria AMORTECIDA)

Saida: mo576_estabilidade.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lambda_rk4(x):
    z = -1j * x
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0


def lambdas_leapfrog(x):
    disc = 1.0 - x ** 2 + 0j
    raiz = np.sqrt(disc)
    lam_fis = -1j * x + raiz
    lam_comp = -1j * x - raiz
    return lam_fis, lam_comp


def lambdas_ab2(x):
    """
    Raizes de lambda^2 - (1 + 3z/2) lambda + z/2 = 0, z = -i x.
    A raiz de sinal '+' na formula de Bhaskara e a raiz FISICA (tende a
    1 quando x->0, e aproxima exp(-ix) para x pequeno); a de sinal '-'
    e a raiz ESPURIA (tende a 0 quando x->0).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    z = -1j * x
    coef_a = 1.0
    coef_b = -(1.0 + 1.5 * z)
    coef_c = 0.5 * z
    disc = np.asarray(coef_b ** 2 - 4 * coef_a * coef_c, dtype=complex)
    raiz = np.sqrt(disc)
    lam1 = (-coef_b + raiz) / (2 * coef_a)
    lam2 = (-coef_b - raiz) / (2 * coef_a)

    # Identificacao da raiz fisica por CONTINUIDADE ao longo do vetor x
    # (em vez de comparar a distancia a exp(-ix) ponto a ponto, o que
    # pode trocar de raiz de forma artificial numa regiao onde ambas
    # as raizes ja estao longe do circulo unitario, ja instaveis).
    # Em x->0, lambda_fis -> 1 e lambda_comp -> 0; usamos isso como
    # ancora e seguimos a raiz mais proxima do valor anterior a cada
    # passo em x.
    fis = np.empty_like(lam1)
    comp = np.empty_like(lam1)
    fis[0] = lam1[0] if abs(lam1[0] - 1.0) < abs(lam2[0] - 1.0) else lam2[0]
    comp[0] = lam2[0] if fis[0] is lam1[0] else lam1[0]
    comp[0] = lam1[0] + lam2[0] - fis[0]
    for i in range(1, len(x)):
        d1 = abs(lam1[i] - fis[i - 1])
        d2 = abs(lam2[i] - fis[i - 1])
        if d1 <= d2:
            fis[i], comp[i] = lam1[i], lam2[i]
        else:
            fis[i], comp[i] = lam2[i], lam1[i]
    return fis, comp


x = np.linspace(0.001, 2.0, 900)

lam_rk4 = lambda_rk4(x)
lam_lf_fis, lam_lf_comp = lambdas_leapfrog(x)
lam_ab2_fis, lam_ab2_comp = lambdas_ab2(x)

fig, axs = plt.subplots(2, 2, figsize=(13, 11))

# --- (a) |lambda| modo fisico, os tres esquemas ---
ax = axs[0, 0]
ax.plot(x, np.abs(lam_rk4), label="RK4 (unica raiz)", linewidth=2, color="tab:purple")
ax.plot(x, np.abs(lam_lf_fis), label="Leapfrog - modo fisico", linewidth=2, color="tab:blue")
ax.plot(x, np.abs(lam_ab2_fis), label="AB2 - modo fisico", linewidth=2, color="tab:green")
ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="Exato (neutro)")
ax.set_xlabel("x = N * dt")
ax.set_ylabel("|lambda(x)|")
ax.set_title("(a) Modulo do fator de amplificacao -- modo fisico")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0.85, 1.5)

# --- (b) |lambda| do modo espurio: Leapfrog (neutro) vs AB2 (amortecido) ---
ax = axs[0, 1]
ax.plot(x, np.abs(lam_lf_comp), label="Leapfrog - modo computacional", linewidth=2,
        color="tab:red", linestyle="--")
ax.plot(x, np.abs(lam_ab2_comp), label="AB2 - raiz espuria", linewidth=2,
        color="tab:orange", linestyle="--")
ax.axhline(1.0, color="k", linestyle=":", linewidth=1)
ax.axvline(1.0, color="gray", linestyle="-.", linewidth=1, label="limite N dt = 1 (Leapfrog)")
ax.set_xlabel("x = N * dt")
ax.set_ylabel("|lambda_espurio(x)|")
ax.set_title(
    "(b) Modo espurio: Leapfrog permanece em |lambda|=1\n"
    "(nao amortece) vs. AB2, cuja raiz espuria -> 0"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.5)

# --- (c) diagrama no plano complexo ---
ax = axs[1, 0]
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), "k:", linewidth=1, label="Circulo unitario")
ax.plot(lam_rk4.real, lam_rk4.imag, linewidth=2, color="tab:purple", label="RK4")
ax.plot(lam_lf_fis.real, lam_lf_fis.imag, linewidth=2, color="tab:blue", label="Leapfrog (fisico)")
ax.plot(lam_lf_comp.real, lam_lf_comp.imag, linewidth=2, color="tab:red", linestyle="--",
        label="Leapfrog (computacional)")
ax.plot(lam_ab2_fis.real, lam_ab2_fis.imag, linewidth=2, color="tab:green", label="AB2 (fisico)")
ax.plot(lam_ab2_comp.real, lam_ab2_comp.imag, linewidth=2, color="tab:orange", linestyle="--",
        label="AB2 (espurio)")
ax.set_xlabel("Re(lambda)")
ax.set_ylabel("Im(lambda)")
ax.set_title("(c) Trajetorias de lambda(x) no plano complexo\n(x de 0 ate 2.0)")
ax.legend(fontsize=7, loc="lower left")
ax.grid(alpha=0.3)
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal")

# --- (d) erro de fase relativo do modo fisico ---
ax = axs[1, 1]
fase_exata = -x
for nome, lam, cor in [
    ("RK4", lam_rk4, "tab:purple"),
    ("Leapfrog (fisico)", lam_lf_fis, "tab:blue"),
    ("AB2 (fisico)", lam_ab2_fis, "tab:green"),
]:
    fase_num = np.unwrap(np.angle(lam))
    erro_fase = (fase_num - fase_exata) / np.where(np.abs(fase_exata) > 1e-12, fase_exata, 1.0)
    ax.plot(x, erro_fase, label=nome, linewidth=2, color=cor)
ax.axhline(0.0, color="k", linestyle=":", linewidth=1)
ax.set_xlabel("x = N * dt")
ax.set_ylabel("(fase numerica - fase exata) / fase exata")
ax.set_title("(d) Erro de fase relativo, modo fisico")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(-0.3, 0.3)

fig.suptitle(
    "MET-576-4 - Estabilidade linear: Leapfrog x Adams-Bashforth-2 x RK4\n"
    "Sistema teste: oscilador de empuxo (coluna oceanica idealizada, Boussinesq linearizado)",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("mo576_estabilidade.png", dpi=150)
print("Figura salva: mo576_estabilidade.png")

print()
print("Resumo em x = N*dt = 0.5:")
x0 = 0.5
lf_fis0, lf_comp0 = lambdas_leapfrog(x0)
ab2_fis0, ab2_comp0 = lambdas_ab2(x0)
ab2_fis0, ab2_comp0 = ab2_fis0[0], ab2_comp0[0]
print(f"  RK4               modo fisico  : |lambda| = {abs(lambda_rk4(x0)):.4f}")
print(f"  Leapfrog          modo fisico  : |lambda| = {abs(lf_fis0):.4f}")
print(f"  Leapfrog          modo comput. : |lambda| = {abs(lf_comp0):.4f}  <-- NAO amortece")
print(f"  Adams-Bashforth-2 modo fisico  : |lambda| = {abs(ab2_fis0):.4f}")
print(f"  Adams-Bashforth-2 raiz espuria : |lambda| = {abs(ab2_comp0):.4f}  <-- amortece")
