# -*- coding: utf-8 -*-
"""
MET-576-4 - Estabilidade e dissipacao em esquemas de integracao temporal
ENTREGAVEL DIFERENCIAL: demonstracao numerica do modo computacional
espurio do Leapfrog e do efeito do filtro de Asselin (Robert-Asselin),
no modelo da coluna oceanica idealizada (oscilador de empuxo).

Por que o modo computacional so aparece no Leapfrog?
-----------------------------------------------------
RK4 e Adams-Bashforth-2, embora um seja de 1 passo e o outro de 2
niveis, tem cada um sua propria estrutura de raizes: RK4 tem uma unica
raiz (sem modo espurio possivel); o AB2 tem uma raiz espuria, mas ela
tende a ZERO quando N dt -> 0 (ver oceano_coluna_estabilidade.py), ou
seja, e' amortecida naturalmente a cada passo. Ja o Leapfrog tem uma
raiz espuria com |lambda_comp| = 1 EXATAMENTE (neutra, nao amortece por
si so) -- por isso e' o UNICO dos tres esquemas deste trabalho em que o
modo computacional persiste indefinidamente sem uma intervencao externa
(o filtro de Asselin).

Estrategia de demonstracao
--------------------------
Para EXCITAR deliberadamente o modo computacional, o nivel de tempo
n-1 usado para iniciar o Leapfrog recebe uma pequena perturbacao
artificial (alem da inicializacao por RK4, que sozinha ja deixaria o
modo computacional quase invisivel). Isso projeta uma componente finita
sobre o modo espurio logo no primeiro passo, tornando o ruido "2 dt"
visivel nas figuras.

Saidas:
  mo576_modo_computacional.png   - serie de w(t): Leapfrog bruto (com
                                    ruido 2dt) vs. AB2 vs. RK4, todos
                                    com a MESMA perturbacao inicial
  mo576_filtro_asselin.png       - efeito de diferentes alpha do filtro
                                    de Asselin sobre o ruido 2dt e sobre
                                    o modo fisico (grafico de energia)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oceano_coluna_schemes import (
    solucao_analitica, energia, rk4, adams_bashforth2, inicializar_ab2,
    leapfrog, inicializar_leapfrog, robert_asselin_filtro, derivada,
)

# ---------------------------------------------------------------------
# Parametros: janela curta (poucos dias), N dt pequeno para ficar bem
# dentro da faixa estavel de todos os esquemas (inclusive o AB2, que e'
# fracamente instavel para qualquer N dt > 0 -- ver nota no script
# oceano_coluna_energia.py)
# ---------------------------------------------------------------------
N = 0.01
dt = 20.0  # N dt = 0.2 -- estavel para RK4/Leapfrog E para o AB2 (que e'
           # fracamente instavel para qualquer N dt > 0, mas com
           # crescimento pequeno o bastante em poucas centenas de
           # passos nesta faixa para nao dominar a comparacao)
w0, b0 = 0.05, 0.0
perturbacao = 0.01  # perturbacao artificial no nivel n-1 do Leapfrog

# NOTA: esta demonstracao usa um dt maior e uma janela CURTA em numero
# de passos (300 passos, ~29 periodos de empuxo) -- deliberadamente
# diferente da simulacao de 60 dias em oceano_coluna_energia.py. O
# motivo: o filtro de Asselin, aplicado a CADA passo, tem um efeito
# cumulativo que cresce com o NUMERO DE PASSOS, nao com o tempo fisico
# em si. Numa janela de milhares de passos (como a de 60 dias com
# dt=5s), mesmo alpha=0.01 amortece o sinal quase por completo (ver
# oceano_coluna_energia.py). Para demonstrar o efeito classico do
# filtro -- supressao do ruido 2dt COM amortecimento discreto e
# perceptivel do modo fisico -- de forma didaticamente clara, usa-se
# aqui uma janela curta (~300 passos), compativel com o numero de
# passos tipico usado na literatura para essa demonstracao.
n_passos = 300
t = np.arange(n_passos + 1) * dt
t_h = t / 3600.0

w_exato, b_exato = solucao_analitica(t, w0, b0, N)


def integrar_leapfrog_perturbado(alpha=None):
    """Leapfrog com o nivel n-1 perturbado, para excitar o modo computacional."""
    w = np.zeros(n_passos + 1)
    b = np.zeros(n_passos + 1)
    w[0], b[0] = w0, b0

    w_m1_exato, b_m1_exato = inicializar_leapfrog(w0, b0, N, dt, modo="rk4")
    w_m1 = w_m1_exato + perturbacao
    b_m1 = b_m1_exato

    w_filt_ant, b_filt_ant = w_m1, b_m1

    for n in range(n_passos):
        if alpha is None:
            w_ant = w_m1 if n == 0 else w[n - 1]
            b_ant = b_m1 if n == 0 else b[n - 1]
        else:
            w_ant = w_filt_ant
            b_ant = b_filt_ant

        w_novo, b_novo = leapfrog(w_ant, b_ant, w[n], b[n], N, dt)

        if alpha is not None:
            w_filt = robert_asselin_filtro(w[n], w_novo, w_filt_ant, alpha)
            b_filt = robert_asselin_filtro(b[n], b_novo, b_filt_ant, alpha)
            w_filt_ant, b_filt_ant = w_filt, b_filt
            w[n], b[n] = w_filt, b_filt

        w[n + 1], b[n + 1] = w_novo, b_novo

    return w, b


# ---------------------------------------------------------------------
# Figura 1: Leapfrog bruto vs. AB2 vs. RK4, todos com a MESMA
# perturbacao artificial no "nivel n-1" (para AB2 e RK4, isso significa
# simplesmente comecar a integracao a partir do estado perturbado
# w0+perturbacao no lugar do estado exato -- ambos sao de partida
# unica/1 nivel, entao nao ha "modo computacional" a excitar neles;
# a perturbacao apenas desloca a condicao inicial)
# ---------------------------------------------------------------------
w_lf_bruto, b_lf_bruto = integrar_leapfrog_perturbado(alpha=None)

w0_pert = w0 + perturbacao  # mesma perturbacao aplicada a AB2/RK4, mas
                             # como deslocamento da condicao inicial

w_rk4 = np.zeros(n_passos + 1)
b_rk4 = np.zeros(n_passos + 1)
w_rk4[0], b_rk4[0] = w0_pert, b0
for n in range(n_passos):
    w_rk4[n + 1], b_rk4[n + 1] = rk4(w_rk4[n], b_rk4[n], N, dt)

w_ab2 = np.zeros(n_passos + 1)
b_ab2 = np.zeros(n_passos + 1)
w_ab2[0], b_ab2[0] = w0_pert, b0
dw_ant, db_ant = inicializar_ab2(w0_pert, b0, N, dt, modo="rk4")
for n in range(n_passos):
    w_ab2[n + 1], b_ab2[n + 1], dw_ant, db_ant = adams_bashforth2(
        w_ab2[n], b_ab2[n], dw_ant, db_ant, N, dt
    )

fig, axs = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

ax = axs[0]
ax.plot(t_h, w_exato, "k--", linewidth=1.2, label="Exato (sem perturbacao)")
ax.plot(t_h, w_lf_bruto, color="tab:red", linewidth=0.9,
        label="Leapfrog (n-1 perturbado, SEM filtro de Asselin)")
ax.set_ylabel("w (m/s)")
ax.set_title(
    "(a) Leapfrog: nivel n-1 perturbado excita o modo computacional -- "
    "ruido \"2 dt\" claramente visivel"
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axs[1]
ax.plot(t_h, w_exato, "k--", linewidth=1.2, label="Exato (sem perturbacao)")
ax.plot(t_h, w_ab2, color="tab:green", linewidth=0.9,
        label="Adams-Bashforth-2 (mesma perturbacao, como deslocamento inicial)")
ax.set_ylabel("w (m/s)")
ax.set_title(
    "(b) AB2 com a mesma perturbacao: nenhum ruido 2 dt (nao ha modo "
    "computacional neutro para excitar)"
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axs[2]
ax.plot(t_h, w_exato, "k--", linewidth=1.2, label="Exato (sem perturbacao)")
ax.plot(t_h, w_rk4, color="tab:purple", linewidth=0.9,
        label="RK4 (mesma perturbacao, como deslocamento inicial)")
ax.set_xlabel("Tempo (horas)")
ax.set_ylabel("w (m/s)")
ax.set_title(
    "(c) RK4 com a mesma perturbacao: nenhum ruido 2 dt (esquema de 1 "
    "passo, sem terceiro nivel de tempo)"
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle(
    "MET-576-4 - Entregavel diferencial: modo computacional do Leapfrog\n"
    "(ausente em AB2 e RK4 com a mesma perturbacao inicial)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("mo576_modo_computacional.png", dpi=150)
print("Figura salva: mo576_modo_computacional.png")
plt.close(fig)

# ---------------------------------------------------------------------
# Figura 2: efeito de diferentes alpha do filtro de Asselin
# ---------------------------------------------------------------------
alphas = [0.0, 0.01, 0.05, 0.1, 0.2]
E0, _, _ = energia(w0, b0, N)

fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axs[0]
ax.plot(t_h, w_exato, "k--", linewidth=1.2, label="Exato")
for a in alphas:
    alpha_arg = None if a == 0.0 else a
    w_a, b_a = integrar_leapfrog_perturbado(alpha=alpha_arg)
    rotulo = "sem filtro (alpha=0)" if a == 0.0 else f"alpha = {a}"
    ax.plot(t_h, w_a, linewidth=0.9, label=rotulo)
ax.set_xlabel("Tempo (horas)")
ax.set_ylabel("w (m/s)")
ax.set_title("(a) Serie de w(t): alpha maior suprime\no ruido 2 dt mais rapido")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axs[1]
for a in alphas:
    alpha_arg = None if a == 0.0 else a
    w_a, b_a = integrar_leapfrog_perturbado(alpha=alpha_arg)
    E_a, _, _ = energia(w_a, b_a, N)
    rotulo = "sem filtro (alpha=0)" if a == 0.0 else f"alpha = {a}"
    ax.plot(t_h, E_a / E0, linewidth=1.2, label=rotulo)
ax.axhline(1.0, color="k", linestyle=":", linewidth=1)
ax.set_xlabel("Tempo (horas)")
ax.set_ylabel("E(t) / E(0)")
ax.set_title(
    "(b) Custo do filtro: alpha maior suprime o modo\n"
    "computacional mas amortece tambem o modo fisico"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(
    "MET-576-4 - Efeito do filtro de Asselin: supressao do modo\n"
    f"computacional vs. amortecimento do modo fisico (janela de {n_passos} passos, "
    f"~{t_h[-1]:.1f} h, ~{t_h[-1]*3600/(2*np.pi/N):.0f} periodos de empuxo)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig("mo576_filtro_asselin.png", dpi=150)
print("Figura salva: mo576_filtro_asselin.png")
plt.close(fig)

print()
print(f"Resumo (energia relativa ao final de {n_passos} passos, ~{t_h[-1]:.2f} h, "
      "mesma perturbacao inicial):")
for a in alphas:
    alpha_arg = None if a == 0.0 else a
    w_a, b_a = integrar_leapfrog_perturbado(alpha=alpha_arg)
    E_a, _, _ = energia(w_a[-1], b_a[-1], N)
    rotulo = "sem filtro" if a == 0.0 else f"alpha={a}"
    print(f"  Leapfrog, {rotulo:14s}: E/E0 = {E_a/E0:.4f}")
E_ab2, _, _ = energia(w_ab2[-1], b_ab2[-1], N)
E_rk4, _, _ = energia(w_rk4[-1], b_rk4[-1], N)
print(f"  AB2 (mesma perturbacao)   : E/E0 = {E_ab2/E0:.4f}")
print(f"  RK4 (mesma perturbacao)   : E/E0 = {E_rk4/E0:.4f}")
