"""
================================================================================
CONSISTENCIA NUMERICA NA TROCA DE FLUXOS EM SISTEMAS OCEANO-ATMOSFERA ACOPLADOS
Modelo de coluna idealizado (2 caixas) inspirado em Manabe & Bryan (1969)
================================================================================

Sistema fisico
--------------
Duas "caixas" de capacidade termica C_a (coluna atmosferica) e C_o (camada de
mistura oceanica, 50 m, seguindo Manabe & Bryan 1969), trocando um fluxo
turbulento linearizado F = lambda*(Ta - To) [W/m2], cada uma amortecida por um
termo radiativo linear (feedback de Planck):

    Ca dTa/dt = -Ba*Ta - lambda*(Ta - To)          (atmosfera)
    Co dTo/dt = -Bo*To + lambda*(Ta - To)          (oceano)

Ta, To sao ANOMALIAS de temperatura (K) em relacao ao equilibrio radiativo.
O termo de troca lambda*(Ta-To) e, por construcao, EXATAMENTE conservativo:
o que sai da atmosfera entra no oceano a cada instante. Isso e importante -
a demonstracao deste estudo NAO e sobre "vazamento" de energia no sentido de
uma assimetria no fluxo trocado, e sim sobre o ERRO SISTEMATICO DE TRAJETORIA
(e, portanto, na energia total simulada) introduzido quando o acoplamento
entre os dois modelos e feito com um passo de tempo de troca (Δt_acopl) maior
que o passo interno de integracao - exatamente a situacao descrita em
Manabe & Bryan (1969): "the atmosphere on 0th, 0.5th and 1st atmospheric year
interacts with the ocean on 0th, 50th and 100th oceanic year".

Como o sistema e linear e de coeficientes constantes, a solucao analitica
exata (autovalores/autovetores) esta disponivel e serve de referencia
("verdade") livre de qualquer erro numerico - permitindo isolar de forma
limpa o erro devido UNICAMENTE ao esquema/frequencia de acoplamento.

Dois esquemas de acoplamento assincrono sao comparados, ambos com a
atmosfera integrada exatamente (solucao fechada local) em subpassos finos
dentro de cada janela de acoplamento, mas usando informacao do oceano de
formas diferentes:

  Esquema A - "acoplamento defasado" (zeroth-order hold):
      O(t) usado pela atmosfera durante toda a janela [t_n, t_n+Δt] e
      mantido CONGELADO no valor To(t_n) (ultimo dado recebido do oceano).
      Equivalente, para a variavel lenta (oceano), a um passo de Euler
      explicito de tamanho Δt_acopl -> erro local O(Δt^2), erro global O(Δt).

  Esquema B - "acoplamento com extrapolacao linear" (preditor-corretor):
      Um passo preditor (identico ao Esquema A) fornece uma estimativa
      To*(t_n+Δt). Dentro da janela, o oceano "visto" pela atmosfera varia
      LINEARMENTE entre To(t_n) e To*(t_n+Δt) (extrapolacao de 1a ordem no
      tempo, tipo Heun/trapezoidal). O fluxo entregue ao oceano ao final da
      janela e a media do fluxo ao longo da janela (nao apenas o valor
      inicial) -> erro local O(Δt^3), erro global O(Δt^2).

Para cada Δt_acopl, calcula-se o erro RMS de Ta, To e da energia total
E' = Ca*Ta + Co*To ao longo de toda a integracao, em relacao a solucao
analitica exata. A lei de potencia erro ~ Δt^p e ajustada em escala log-log;
o resultado esperado (e verificado numericamente abaixo) e p_A ~ 1 e p_B ~ 2,
i.e., o acoplamento infrequente introduz um erro sistematico EQUIVALENTE, em
ordem de convergencia, a um esquema de integracao temporal de 1a ordem
(Euler) - o analogo, no tempo de acoplamento, do erro de truncamento espacial
associado ao numero de Courant (CFL) em esquemas advectivos.

Autor: material de apoio - MET-579 (Grupo: acoplamento oceano-atmosfera)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. PARAMETROS FISICOS
# ------------------------------------------------------------------
SEC_DAY = 86400.0

C_a = 1.0e7            # J/m2/K  - capacidade termica da coluna atmosferica
                        #   (~ cp_ar * ps/g = 1004 * 1e5/9.8 ~ 1.0e7 J/m2/K)
C_o = 2.1e8             # J/m2/K  - capacidade termica da camada de mistura
                        #   oceanica de 50 m (rho_agua*cp_agua*h),
                        #   seguindo a profundidade usada em Manabe & Bryan (1969)
lam = 20.0              # W/m2/K  - coeficiente de troca turbulenta (calor
                        #   sensivel + latente linearizado)

tau_a_rad = 365.0 * SEC_DAY          # relaxacao radiativa da atmosfera isolada ~ 1 ano
tau_o_rad = 100.0 * 365.0 * SEC_DAY  # relaxacao radiativa do oceano isolado ~ 100 anos
                                     # (consistente com a nota de M&B: "thermal
                                     #  relaxation time of the ocean is of the
                                     #  order of centuries")

B_a = C_a / tau_a_rad    # W/m2/K
B_o = C_o / tau_o_rad    # W/m2/K

# Matriz do sistema linear acoplado: d/dt [Ta,To] = M @ [Ta,To]
M = np.array([
    [-(B_a + lam) / C_a,        lam / C_a],
    [       lam / C_o,   -(B_o + lam) / C_o],
])

# condicao inicial: pulso de aquecimento atmosferico de 1 K, oceano em repouso
x0 = np.array([1.0, 0.0])

TOTAL_DAYS = 3650.0   # 10 anos - suficiente para varias janelas de acoplamento
                        # mesmo nos casos mais grosseiros, sem exigir o
                        # equilibrio completo (~decadas) do modo lento oceanico

# ------------------------------------------------------------------
# 2. SOLUCAO ANALITICA EXATA (referencia / "verdade")
# ------------------------------------------------------------------
eigvals, eigvecs = np.linalg.eig(M)
coeffs = np.linalg.solve(eigvecs, x0)

# timescales dos dois modos (dias) - diagnostico
tau_modes_days = -1.0 / np.real(eigvals) / SEC_DAY
print(f"Escalas de tempo dos modos acoplados: {tau_modes_days} dias")
print(f"  (modo rapido ~ {tau_modes_days.min():.2f} dias, "
      f"modo lento ~ {tau_modes_days.max():.1f} dias "
      f"= {tau_modes_days.max()/365:.1f} anos)")
tau_fast_days = tau_modes_days.min()


def analytic(t_days):
    """Solucao exata [Ta,To] no tempo t_days (escalar ou array)."""
    t = np.atleast_1d(t_days).astype(float) * SEC_DAY
    # shape (2, nt)
    out = np.real(eigvecs @ (coeffs[:, None] * np.exp(np.outer(eigvals, t))))
    return out  # out[0]=Ta(t), out[1]=To(t)


# ------------------------------------------------------------------
# 3. INTEGRACAO EXATA DA ATMOSFERA COM OCEANO "VISTO" PRESCRITO
# ------------------------------------------------------------------
# Dentro de uma janela, com To(t) prescrito (congelado ou linear), a EDO da
# atmosfera e linear com coeficiente constante + forcante conhecida:
#   dTa/dt = a*Ta + b(t),   a = -(Ba+lam)/Ca
# Resolvemos exatamente por variacao de parametros para b(t) constante
# e para b(t) linear em t (evita erro numerico extra do lado atmosferico,
# isolando o erro que vem UNICAMENTE da defasagem/():interpolacao do oceano).

a_coef = -(B_a + lam) / C_a


def integrate_Ta_frozen(Ta0, To_frozen, dt_sec, n_out=50):
    """Ta exata em [0,dt_sec] com To constante = To_frozen.
    Retorna vetor de tempos (n_out+1 pontos) e Ta correspondente,
    para permitir o calculo do fluxo medio na janela."""
    t = np.linspace(0.0, dt_sec, n_out + 1)
    b = (lam / C_a) * To_frozen
    Ta_eq = -b / a_coef
    Ta = Ta_eq + (Ta0 - Ta_eq) * np.exp(a_coef * t)
    return t, Ta


def integrate_Ta_linear_To(Ta0, To0, To1, dt_sec, n_out=50):
    """Ta exata em [0,dt_sec] com To(t) variando linearmente de To0 a To1.
    dTa/dt = a*Ta + (lam/Ca)*(To0 + (To1-To0)*t/dt_sec)
    Solucao fechada para forcante linear em t."""
    t = np.linspace(0.0, dt_sec, n_out + 1)
    k = lam / C_a
    slope = (To1 - To0) / dt_sec
    # particular solution for b(t) = k*(To0 + slope*t) is linear: p(t) = A + B t
    # p'(t) = a*p(t) + k*To0 + k*slope*t  =>  B = a*(A+Bt) + k*To0 + k*slope*t
    # match coefficients: B = a*B + k*slope  => B = k*slope/(1-a)  [careful with a<0]
    # matching t^0: 0*const handled via A: B = a*A + k*To0  => A = (B - k*To0)/a
    Bc = (k * slope) / (-a_coef)  # since a - a*1 identity handled below robustly
    # Direct robust solve: assume particular solution p(t)=A+B*t
    # p' = B ; a*p + k*To0 + k*slope*t = a*A + a*B*t + k*To0 + k*slope*t
    # Match t^1: 0 = a*B + k*slope  -> B = -k*slope/a
    Bc = -k * slope / a_coef
    # Match t^0: B = a*A + k*To0    -> A = (B - k*To0)/a
    Ac = (Bc - k * To0) / a_coef
    homog_coef = Ta0 - (Ac + Bc * 0.0)
    Ta = (Ac + Bc * t) + homog_coef * np.exp(a_coef * t)
    return t, Ta


def ocean_step(Ta_series, To_used_series, To0, dt_sec, To_damp):
    """Um unico passo do oceano (tamanho dt_sec = Delta t_acopl), forcado
    pelo fluxo medio ao longo da janela: F_avg = lam*mean(Ta - To_used).
    O termo de amortecimento radiativo -Bo*To e avaliado em To_damp:
      - Esquema A: To_damp = To0            -> Euler explicito (1a ordem)
      - Esquema B: To_damp = (To0+To*)/2    -> trapezoidal/Crank-Nicolson (2a ordem)
    """
    F_avg = lam * np.mean(Ta_series - To_used_series)
    dTo = (dt_sec / C_o) * (-B_o * To_damp + F_avg)
    return To0 + dTo


# ------------------------------------------------------------------
# 4. ESQUEMAS DE ACOPLAMENTO ASSINCRONO
# ------------------------------------------------------------------
def run_scheme(dt_couple_days, scheme="A", n_sub=60):
    """Integra o sistema acoplado assincronamente de 0 a TOTAL_DAYS com
    passo de acoplamento dt_couple_days, usando o Esquema A (defasado) ou
    B (extrapolacao linear/preditor-corretor). Retorna arrays de tempo,
    Ta, To amostrados ao final de cada janela (mais os subpassos)."""
    dt_c = dt_couple_days * SEC_DAY
    n_windows = int(round(TOTAL_DAYS / dt_couple_days))

    Ta_n, To_n = x0[0], x0[1]
    t_all = [0.0]
    Ta_all = [Ta_n]
    To_all = [To_n]
    t_nodes = [0.0]      # amostras APENAS nos instantes de troca (t_n) - usadas
    Ta_nodes = [Ta_n]    # para o calculo de erro/ordem de convergencia, sem
    To_nodes = [To_n]    # contaminacao pela interpolacao linear usada so p/ plot

    for _ in range(n_windows):
        if scheme == "A":
            t_loc, Ta_series = integrate_Ta_frozen(Ta_n, To_n, dt_c, n_sub)
            To_used_series = np.full_like(Ta_series, To_n)
            To_damp = To_n  # amortecimento radiativo: Euler explicito (1a ordem)
        elif scheme == "B":
            # passo preditor (Esquema A) para estimar To no fim da janela
            _, Ta_pred = integrate_Ta_frozen(Ta_n, To_n, dt_c, n_sub)
            To_used_pred = np.full_like(Ta_pred, To_n)
            To_star = ocean_step(Ta_pred, To_used_pred, To_n, dt_c, To_damp=To_n)
            # passo corretor: To varia linearmente entre To_n e To_star
            t_loc, Ta_series = integrate_Ta_linear_To(Ta_n, To_n, To_star, dt_c, n_sub)
            To_used_series = To_n + (To_star - To_n) * (t_loc / dt_c)
            To_damp = 0.5 * (To_n + To_star)  # amortecimento: trapezoidal (2a ordem)
        else:
            raise ValueError("scheme deve ser 'A' ou 'B'")

        To_np1 = ocean_step(Ta_series, To_used_series, To_n, dt_c, To_damp=To_damp)
        Ta_np1 = Ta_series[-1]

        t_all.extend((t_loc[1:] / SEC_DAY + t_all[-1]).tolist())
        Ta_all.extend(Ta_series[1:].tolist())
        To_all.extend((To_n + (To_np1 - To_n) * (t_loc[1:] / dt_c)).tolist())

        Ta_n, To_n = Ta_np1, To_np1
        t_nodes.append(t_all[-1])
        Ta_nodes.append(Ta_n)
        To_nodes.append(To_n)

    return (np.array(t_all), np.array(Ta_all), np.array(To_all),
            np.array(t_nodes), np.array(Ta_nodes), np.array(To_nodes))


# ------------------------------------------------------------------
# 5. EXPERIMENTO DE CONVERGENCIA: ERRO vs Δt_acopl
# ------------------------------------------------------------------
dt_couple_list = np.array([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256])  # dias

results = {"A": {"Ta": [], "To": [], "E": []},
           "B": {"Ta": [], "To": [], "E": []}}

for scheme in ("A", "B"):
    for dtc in dt_couple_list:
        _, _, _, t_days, Ta_num, To_num = run_scheme(dtc, scheme=scheme)
        ref = analytic(t_days)
        Ta_ref, To_ref = ref[0], ref[1]

        rmse_Ta = np.sqrt(np.mean((Ta_num - Ta_ref) ** 2))
        rmse_To = np.sqrt(np.mean((To_num - To_ref) ** 2))

        E_num = C_a * Ta_num + C_o * To_num
        E_ref = C_a * Ta_ref + C_o * To_ref
        rmse_E = np.sqrt(np.mean((E_num - E_ref) ** 2))  # J/m2

        results[scheme]["Ta"].append(rmse_Ta)
        results[scheme]["To"].append(rmse_To)
        results[scheme]["E"].append(rmse_E)

for scheme in ("A", "B"):
    for var in ("Ta", "To", "E"):
        results[scheme][var] = np.array(results[scheme][var])


def fit_order(dtc, err, i0=None, i1=None):
    """Ajuste log-log err ~ C*dtc^p -> retorna p (ordem de convergencia),
    opcionalmente restrito ao subconjunto de indices [i0:i1]."""
    logx = np.log(dtc[i0:i1])
    logy = np.log(err[i0:i1])
    p, logC = np.polyfit(logx, logy, 1)
    return p


# Ordem "assintotica": ajuste usando apenas os 5 menores Δt_acopl, todos
# bem menores que a escala de tempo do modo rapido (tau_fast_days) - regime
# em que a expansao em serie de Taylor (base teorica da ordem de convergencia)
# de fato se aplica.
N_ASYMPT = 5
order_A_Ta = fit_order(dt_couple_list, results["A"]["Ta"], 0, N_ASYMPT)
order_B_Ta = fit_order(dt_couple_list, results["B"]["Ta"], 0, N_ASYMPT)
order_A_To = fit_order(dt_couple_list, results["A"]["To"], 0, N_ASYMPT)
order_B_To = fit_order(dt_couple_list, results["B"]["To"], 0, N_ASYMPT)
order_A_E = fit_order(dt_couple_list, results["A"]["E"], 0, N_ASYMPT)
order_B_E = fit_order(dt_couple_list, results["B"]["E"], 0, N_ASYMPT)

# Ordem "faixa completa": todo o intervalo testado (inclui Δt_acopl > tau_fast,
# fora do regime assintotico - a ordem aparente cai por saturacao nao-linear)
order_A_Ta_full = fit_order(dt_couple_list, results["A"]["Ta"])
order_B_Ta_full = fit_order(dt_couple_list, results["B"]["Ta"])
order_A_To_full = fit_order(dt_couple_list, results["A"]["To"])
order_B_To_full = fit_order(dt_couple_list, results["B"]["To"])
order_A_E_full = fit_order(dt_couple_list, results["A"]["E"])
order_B_E_full = fit_order(dt_couple_list, results["B"]["E"])

print(f"\nEscala de tempo do modo rapido: tau_fast = {tau_fast_days:.2f} dias "
      f"(regime assintotico: Δt_acopl << {tau_fast_days:.1f} dias)")
print("\n=== ORDEM DE CONVERGENCIA - REGIME ASSINTOTICO (Δt_acopl <= "
      f"{dt_couple_list[N_ASYMPT-1]:.2f} dias) ===")
print(f"  Ta   -  Esquema A (defasado): p = {order_A_Ta:.2f}   |  Esquema B (linear): p = {order_B_Ta:.2f}")
print(f"  To   -  Esquema A (defasado): p = {order_A_To:.2f}   |  Esquema B (linear): p = {order_B_To:.2f}")
print(f"  E'   -  Esquema A (defasado): p = {order_A_E:.2f}   |  Esquema B (linear): p = {order_B_E:.2f}")
print("\n=== ORDEM DE CONVERGENCIA - FAIXA COMPLETA (0.0625 a 256 dias) ===")
print(f"  Ta   -  Esquema A (defasado): p = {order_A_Ta_full:.2f}   |  Esquema B (linear): p = {order_B_Ta_full:.2f}")
print(f"  To   -  Esquema A (defasado): p = {order_A_To_full:.2f}   |  Esquema B (linear): p = {order_B_To_full:.2f}")
print(f"  E'   -  Esquema A (defasado): p = {order_A_E_full:.2f}   |  Esquema B (linear): p = {order_B_E_full:.2f}")

# ------------------------------------------------------------------
# 6. FIGURAS
# ------------------------------------------------------------------
NAVY = "#0B2545"
GOLD = "#C9A227"
GRAY = "#6E6E6E"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
})

# --- Figura 1: convergencia do erro (log-log) ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
titles = [r"Erro RMS em $T_a'$ (K)", r"Erro RMS em $T_o'$ (K)",
          r"Erro RMS na energia total $E'=C_aT_a'+C_oT_o'$ (J m$^{-2}$)"]
varkeys = ["Ta", "To", "E"]
orders = [(order_A_Ta, order_B_Ta), (order_A_To, order_B_To), (order_A_E, order_B_E)]

for ax, key, title, (pA, pB) in zip(axes, varkeys, titles, orders):
    ax.loglog(dt_couple_list, results["A"][key], "o-", color=NAVY, lw=2, ms=6,
              label=f"Esq. A: defasado ($p\\approx{pA:.2f}$)")
    ax.loglog(dt_couple_list, results["B"][key], "s-", color=GOLD, lw=2, ms=6,
              label=f"Esq. B: extrapolação linear ($p\\approx{pB:.2f}$)")

    # linhas de referencia de 1a e 2a ordem, ancoradas ao primeiro ponto
    # (regime assintotico, Δt << tau_fast)
    ref1 = results["A"][key][0] * (dt_couple_list / dt_couple_list[0]) ** 1
    ref2 = results["B"][key][0] * (dt_couple_list / dt_couple_list[0]) ** 2
    ax.loglog(dt_couple_list, ref1, "--", color=GRAY, lw=1.2, label=r"refer. $\Delta t^{1}$")
    ax.loglog(dt_couple_list, ref2, ":", color=GRAY, lw=1.2, label=r"refer. $\Delta t^{2}$")

    ax.axvline(tau_fast_days, color="firebrick", lw=1.0, ls="-.", alpha=0.7)
    ax.text(tau_fast_days * 1.15, results["A"][key][-1] * 0.15,
            r"$\tau_{r\acute{a}pido}$", color="firebrick", fontsize=8, rotation=90)

    ax.set_xlabel(r"$\Delta t_{acopl}$ (dias)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")

fig.suptitle("Convergência do erro de acoplamento em função do passo de troca de fluxos",
             fontsize=13, color=NAVY, y=1.04)
fig.tight_layout()
fig.savefig("fig1_convergencia_erro.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- Figura 2: erro sistematico (numerico - analitico) ao longo do tempo ---
# Como o erro de acoplamento e muito pequeno frente ao sinal (K), as
# trajetorias brutas ficam visualmente indistinguiveis da referencia; o
# vies sistematico so fica visivel plotando o ERRO (numerico - analitico)
# diretamente, o que tambem evidencia seu carater CUMULATIVO (nao apenas
# ruido aleatorio de alta frequencia).
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for ax, dtc in zip(axes, [16.0, 256.0]):
    tA, TaA, ToA, _, _, _ = run_scheme(dtc, "A")
    tB, TaB, ToB, _, _, _ = run_scheme(dtc, "B")
    refA = analytic(tA)
    refB = analytic(tB)

    ax.plot(tA / 365, (ToA - refA[1]) * 1000, "-", color=NAVY, lw=1.6,
            label=r"Esq. A: defasado")
    ax.plot(tB / 365, (ToB - refB[1]) * 1000, "-", color=GOLD, lw=1.6,
            label=r"Esq. B: extrapolação linear")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("tempo (anos)")
    ax.set_ylabel(r"erro em $T_o'$ (mK)")
    ax.set_title(rf"$\Delta t_{{acopl}} = {dtc:.0f}$ dias", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Erro sistemático (numérico − analítico) da anomalia oceânica: caráter cumulativo do viés de acoplamento",
             fontsize=12.5, color=NAVY, y=1.03)
fig.tight_layout()
fig.savefig("fig2_trajetorias.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- Figura 3: viés sistemático de energia final (equivalente a forcante espúria) ---
# energia total ao final da integracao (10 anos) - erro absoluto expresso
# como uma "forcante equivalente" media (W/m2) que produziria o mesmo desvio
years_end = TOTAL_DAYS * SEC_DAY
fig, ax = plt.subplots(figsize=(6.8, 4.8))
for scheme, color, marker in [("A", NAVY, "o"), ("B", GOLD, "s")]:
    bias_Wm2 = results[scheme]["E"] / years_end  # J/m2 dividido por s = W/m2 equivalente medio
    ax.loglog(dt_couple_list, bias_Wm2, marker + "-", color=color, lw=2, ms=6,
              label=f"Esquema {scheme}")
ax.set_xlabel(r"$\Delta t_{acopl}$ (dias)")
ax.set_ylabel(r"viés equivalente de energia (W m$^{-2}$)")
ax.set_title("Viés sistemático equivalente a uma forçante radiativa espúria\n(erro de energia acumulado / 10 anos)",
             fontsize=11, color=NAVY)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("fig3_vies_energetico.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("\nFiguras salvas: fig1_convergencia_erro.png, fig2_trajetorias.png, fig3_vies_energetico.png")

# ------------------------------------------------------------------
# 7. TABELA-RESUMO (para inclusao no relatorio)
# ------------------------------------------------------------------
print("\n=== TABELA: erro RMS de To (K) por passo de acoplamento ===")
print(f"{'Δt_acopl (dias)':>16} | {'Esq. A (defasado)':>18} | {'Esq. B (linear)':>16}")
for i, dtc in enumerate(dt_couple_list):
    print(f"{dtc:16.1f} | {results['A']['To'][i]:18.3e} | {results['B']['To'][i]:16.3e}")

np.savez("resultados_convergencia.npz",
         dt_couple_list=dt_couple_list,
         A_Ta=results["A"]["Ta"], A_To=results["A"]["To"], A_E=results["A"]["E"],
         B_Ta=results["B"]["Ta"], B_To=results["B"]["To"], B_E=results["B"]["E"])
print("\nResultados numericos salvos em resultados_convergencia.npz")
