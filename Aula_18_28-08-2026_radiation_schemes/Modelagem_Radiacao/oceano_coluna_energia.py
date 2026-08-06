# -*- coding: utf-8 -*-
"""
MET-576-4 - Estabilidade e dissipacao em esquemas de integracao temporal
Simulacao idealizada (coluna oceanica, oscilador de empuxo) comparando
Leapfrog, Adams-Bashforth-2 e RK4 quanto a deriva de energia mecanica
ao longo de uma integracao longa (60 dias, dentro da janela de 30-90
dias pedida no enunciado).

Parametros fisicos default:
  N = 0.01 s^-1  (frequencia de Brunt-Vaisala tipica de termoclina
                  oceanica; periodo de empuxo T = 2 pi / N ~ 10.5 min)
  60 dias de integracao ~ 4110 periodos de empuxo -- uma integracao
  longa o bastante para que erros pequenos por passo se acumulem em
  deriva de energia claramente mensuravel, o que e o ponto central do
  diagnostico pedido.

dt = 60 s (N dt = 0.6, dentro da faixa estavel de todos os esquemas
testados, mas grande o suficiente para tornar a deriva visivel em
poucos dias de figura).

Saidas:
  mo576_energia_deriva.png     - E(t)/E(0) ao longo de 60 dias, os 3
                                 esquemas + Leapfrog com filtro RAF
  mo576_energia_vs_dt.png      - deriva de energia ao final da
                                 integracao, em funcao de dt (para
                                 varios N dt), destacando o custo
                                 computacional (avaliacoes de f por
                                 passo) de cada esquema
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oceano_coluna_schemes import (
    solucao_analitica, energia, rk4, adams_bashforth2, inicializar_ab2,
    leapfrog, inicializar_leapfrog, robert_asselin_filtro,
)

# ---------------------------------------------------------------------
# Parametros da integracao principal
# ---------------------------------------------------------------------
N = 0.01                 # s^-1
T_empuxo = 2 * np.pi / N  # periodo de empuxo, s
dt = 5.0                  # s  (N dt = 0.05)
# NOTA IMPORTANTE (achado do trabalho): o Adams-Bashforth-2 aplicado a
# este oscilador puro (autovalor imaginario puro, sem amortecimento
# fisico algum) e' FRACAMENTE INSTAVEL para qualquer N dt > 0 (ver
# oceano_coluna_estabilidade.py, raiz fisica com |lambda| > 1 sempre).
# Para N dt = 0.6 (a escolha original, ok para RK4/Leapfrog) essa
# instabilidade fraca ja causa overflow numerico bem antes de 60 dias
# de integracao (o crescimento por passo, ~2.7% em N dt=0.5, se
# acumula exponencialmente ao longo de ~86000 passos). dt = 5 s
# (N dt = 0.05) foi escolhido para que a instabilidade do AB2 permaneca
# GRADUAL e MENSURAVEL ao longo de 60 dias (fator de crescimento total
# esperado ~5x em amplitude, ~25x em energia) sem estourar
# numericamente -- o que e exatamente o comportamento que o diagnostico
# de deriva de energia deste trabalho deve evidenciar.
dias_simulacao = 60
t_fim = dias_simulacao * 86400.0
n_passos = int(round(t_fim / dt))

w0, b0 = 0.05, 0.0  # velocidade vertical inicial, m/s (tipica de onda interna)

t = np.arange(n_passos + 1) * dt
t_dias = t / 86400.0

print(f"N dt = {N*dt:.3f}")
print(f"Periodo de empuxo: {T_empuxo/60.0:.2f} min; "
      f"{dias_simulacao} dias = {t_fim/T_empuxo:.0f} periodos de empuxo, "
      f"{n_passos} passos de dt = {dt:.0f} s")

E0, KE0, APE0 = energia(w0, b0, N)


# ---------------------------------------------------------------------
# Integracao RK4
# ---------------------------------------------------------------------
def integrar_rk4(n_passos, dt):
    w = np.zeros(n_passos + 1)
    b = np.zeros(n_passos + 1)
    w[0], b[0] = w0, b0
    for n in range(n_passos):
        w[n + 1], b[n + 1] = rk4(w[n], b[n], N, dt)
    return w, b


# ---------------------------------------------------------------------
# Integracao Adams-Bashforth-2
# ---------------------------------------------------------------------
def integrar_ab2(n_passos, dt):
    w = np.zeros(n_passos + 1)
    b = np.zeros(n_passos + 1)
    w[0], b[0] = w0, b0
    dw_ant, db_ant = inicializar_ab2(w0, b0, N, dt, modo="rk4")
    for n in range(n_passos):
        w[n + 1], b[n + 1], dw_ant, db_ant = adams_bashforth2(
            w[n], b[n], dw_ant, db_ant, N, dt
        )
    return w, b


# ---------------------------------------------------------------------
# Integracao Leapfrog, com e sem filtro de Robert-Asselin
# ---------------------------------------------------------------------
def integrar_leapfrog(n_passos, dt, alpha=None):
    w = np.zeros(n_passos + 1)
    b = np.zeros(n_passos + 1)
    w_m1, b_m1 = inicializar_leapfrog(w0, b0, N, dt, modo="rk4")
    w[0], b[0] = w0, b0

    w_filt_ant, b_filt_ant = w_m1, b_m1
    for n in range(n_passos):
        w_ant = w_filt_ant if alpha is not None else (w_m1 if n == 0 else w[n - 1])
        b_ant = b_filt_ant if alpha is not None else (b_m1 if n == 0 else b[n - 1])

        w_novo, b_novo = leapfrog(w_ant, b_ant, w[n], b[n], N, dt)

        if alpha is not None:
            w_filt = robert_asselin_filtro(w[n], w_novo, w_filt_ant, alpha)
            b_filt = robert_asselin_filtro(b[n], b_novo, b_filt_ant, alpha)
            w_filt_ant, b_filt_ant = w_filt, b_filt
            w[n], b[n] = w_filt, b_filt

        w[n + 1], b[n + 1] = w_novo, b_novo
    return w, b


w_rk4, b_rk4 = integrar_rk4(n_passos, dt)
w_ab2, b_ab2 = integrar_ab2(n_passos, dt)
w_lf, b_lf = integrar_leapfrog(n_passos, dt, alpha=None)
w_lf_raf, b_lf_raf = integrar_leapfrog(n_passos, dt, alpha=0.0001)
# NOTA IMPORTANTE (achado do trabalho): valores tipicos operacionais de
# alpha (0.01-0.05) aplicados a CADA passo ao longo de ~1 milhao de
# passos (60 dias com dt=5s) amortecem o modo fisico quase que
# totalmente (colapso de amplitude para menos de 1% do valor original
# ja em poucos dias -- testado e confirmado numericamente). Isso NAO e'
# um erro de implementacao: e' uma consequencia matematica conhecida do
# filtro de Asselin classico quando aplicado de forma continua por
# integracoes muito longas (ver Amezcua et al. 2011; Williams 2009 -
# filtro RAW como alternativa). Por isso, para tornar a deriva do modo
# fisico visivel e gradual ao longo de 60 dias inteiros, usa-se aqui um
# alpha muito menor (0.0001) do que o tipico operacional -- e o proprio
# contraste entre este valor e o valor operacional tipico e' parte do
# resultado a ser discutido no relatorio.

resultados = {
    "RK4": (w_rk4, b_rk4),
    "Adams-Bashforth-2": (w_ab2, b_ab2),
    "Leapfrog (sem filtro)": (w_lf, b_lf),
    "Leapfrog (filtro RAF, alpha=0.0001)": (w_lf_raf, b_lf_raf),
}

# ---------------------------------------------------------------------
# Figura 1: deriva de energia normalizada ao longo de 60 dias
# ---------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

ax = axs[0]
for nome, (w, b) in resultados.items():
    E, KE, APE = energia(w, b, N)
    ax.plot(t_dias, E / E0, linewidth=1.2, label=nome)
ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="Exato (conservado)")
ax.set_yscale("log")
ax.set_ylabel("E(t) / E(0)  (escala log)")
ax.set_title(
    f"(a) Deriva de energia mecanica total, {dias_simulacao} dias (N dt = {N*dt:.2f})\n"
    "AB2 diverge por ~25x -- instabilidade fraca real, nao artefato numerico"
)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3, which="both")

ax = axs[1]
# Painel (b) exclui o AB2 de proposito: sua divergencia de ~2500% domina
# a escala e esconde as diferencas, muito menores, entre RK4 e as duas
# variantes do Leapfrog -- que sao o ponto central deste painel.
for nome, (w, b) in resultados.items():
    if nome == "Adams-Bashforth-2":
        continue
    E, KE, APE = energia(w, b, N)
    ax.plot(t_dias, (E / E0 - 1.0) * 100, linewidth=1.4, label=nome)
ax.axhline(0.0, color="k", linestyle=":", linewidth=1)
ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Deriva de energia (%)")
ax.set_title(
    "(b) RK4 e Leapfrog (com e sem filtro RAF), sem o AB2\n"
    "(AB2 omitido aqui de proposito -- sua escala de ~2500% esconderia estas curvas)"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(
    "MET-576-4 - Diagnostico de conservacao de energia: Leapfrog, AB2, RK4\n"
    "Modelo: oscilador de empuxo (coluna oceanica idealizada, Boussinesq linearizado)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("mo576_energia_deriva.png", dpi=150)
print("Figura salva: mo576_energia_deriva.png")
plt.close(fig)

# ---------------------------------------------------------------------
# Figura 2: deriva de energia final vs. dt (custo computacional)
# ---------------------------------------------------------------------
dts_teste = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
t_fim_curto = 10 * 86400.0  # 10 dias, fixo, para o teste de sensibilidade a dt
# NOTA: a faixa de dt foi escolhida para manter o AB2 (que e' fracamente
# instavel para QUALQUER N dt > 0 neste problema, ver
# oceano_coluna_estabilidade.py) dentro de uma faixa de crescimento
# mensuravel (ate' ~10x em 10 dias); valores de dt maiores fazem o AB2
# divergir por muitas ordens de grandeza em poucos dias, o que tornaria
# o grafico comparativo ilegivel (RK4 e Leapfrog permanecem estaveis
# em toda essa faixa e bem alem dela).

deriva_rk4 = []
deriva_ab2 = []
deriva_lf = []
avals_por_segundo = {"RK4": [], "Adams-Bashforth-2": [], "Leapfrog": []}

for dt_i in dts_teste:
    n_i = int(round(t_fim_curto / dt_i))

    w_r, b_r = integrar_rk4(n_i, dt_i)
    E_r, _, _ = energia(w_r[-1], b_r[-1], N)
    deriva_rk4.append((E_r / E0 - 1.0) * 100)

    w_a, b_a = integrar_ab2(n_i, dt_i)
    E_a, _, _ = energia(w_a[-1], b_a[-1], N)
    deriva_ab2.append((E_a / E0 - 1.0) * 100)

    w_l, b_l = integrar_leapfrog(n_i, dt_i, alpha=None)
    E_l, _, _ = energia(w_l[-1], b_l[-1], N)
    deriva_lf.append((E_l / E0 - 1.0) * 100)

    # custo: avaliacoes da funcao "derivada" por segundo de tempo simulado
    avals_por_segundo["RK4"].append(4.0 / dt_i)          # RK4: 4 avaliacoes/passo
    avals_por_segundo["Adams-Bashforth-2"].append(1.0 / dt_i)  # AB2: 1 avaliacao/passo
    avals_por_segundo["Leapfrog"].append(1.0 / dt_i)      # Leapfrog: 1 avaliacao/passo (aprox.)

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axs[0]
ax.plot(N * dts_teste, deriva_rk4, "o-", label="RK4", color="tab:purple")
ax.plot(N * dts_teste, deriva_ab2, "o-", label="Adams-Bashforth-2", color="tab:green")
ax.plot(N * dts_teste, deriva_lf, "o-", label="Leapfrog (sem filtro)", color="tab:blue")
ax.axhline(0.0, color="k", linestyle=":", linewidth=1)
ax.set_xlabel("N * dt")
ax.set_ylabel("Deriva de energia em 10 dias (%)")
ax.set_title("(a) Deriva de energia final vs. N dt")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axs[1]
estilos_custo = {"RK4": "o-", "Adams-Bashforth-2": "s--", "Leapfrog": "^:"}
for nome, vals in avals_por_segundo.items():
    ax.plot(N * dts_teste, vals, estilos_custo[nome], label=nome, markersize=7)
ax.set_xlabel("N * dt")
ax.set_ylabel("Avaliacoes de f(w,b) por segundo simulado")
ax.set_title("(b) Custo computacional relativo\n(avaliacoes da funcao por segundo de integracao)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_yscale("log")

fig.suptitle(
    "MET-576-4 - Sensibilidade da deriva de energia e custo computacional vs. N dt",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("mo576_energia_vs_dt.png", dpi=150)
print("Figura salva: mo576_energia_vs_dt.png")
plt.close(fig)

# ---------------------------------------------------------------------
# Resumo numerico
# ---------------------------------------------------------------------
print()
print(f"Deriva de energia ao final de {dias_simulacao} dias (N dt = {N*dt:.2f}):")
for nome, (w, b) in resultados.items():
    E, _, _ = energia(w[-1], b[-1], N)
    print(f"  {nome:38s}: {(E/E0 - 1.0)*100:+9.4f} %")

print()
print("Nota: RK4 custa 4 avaliacoes de f por passo (mais caro por passo),")
print("mas nao precisa de historico de niveis anteriores. AB2 e Leapfrog")
print("custam apenas 1 avaliacao por passo, porem exigem armazenar o")
print("nivel (ou a derivada) anterior -- e, no caso do Leapfrog, exigem")
print("tambem lidar com o modo computacional (filtro de Asselin).")
