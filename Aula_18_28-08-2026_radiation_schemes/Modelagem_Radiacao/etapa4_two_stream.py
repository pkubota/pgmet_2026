"""
ETAPA 4 - Espalhamento: modelo TWO-STREAM completo (nuvens e aerossois)
=========================================================================

Ate a Etapa 3, cada camada so absorvia e emitia (omega_0 = 0, sem
espalhamento) -- adequado para gases, mas incapaz de representar nuvens ou
aerossois, cujo espalhamento (omega_0 proximo de 1) domina o balanco de
radiacao de onda curta.

Esta etapa implementa exatamente a Secao 15 do material do curso:

  1. Solucao fechada de uma camada homogenea (tau, omega_0, g) -- Eqs.
     15.19-15.22 (Exemplos 15.3 e 15.4): espalhamento conservativo e
     nao-conservativo.
  2. O "metodo de adicao" (adding method / interaction principle, Eq.
     15.30-15.32) para combinar varias camadas (por ex. nuvem + gases +
     superficie) numa unica refletancia/transmitancia efetiva da coluna.
  3. Delta-scaling (delta-two-stream, Secao 15.4c) para tratar nuvens com
     forte espalhamento para frente (g alto).
  4. Aplicacao pratica: albedo do sistema nuvem+superficie em funcao da
     espessura optica e do tamanho de particula da nuvem (cf. Fig. 15.4
     do material do curso).

Convencao: D = fator de difusividade (2.0, aproximacao "hemispheric
constant"/quadratura, comum na literatura); b = fracao de retroespalhamento,
aproximada por b = (1-g)/2 (relacao padrao para a funcao de fase de
Henyey-Greenstein).

COMO EXECUTAR
--------------
    python3 etapa4_two_stream.py

Sem argumentos de linha de comando. Os casos de teste (espessuras opticas
de nuvem, tipos de gota, albedos de superficie) estao no
`if __name__ == "__main__":`.

SAIDA: prints com as 3 validacoes (conservacao de energia, limites
fisicos, e o teste de subdivisao em N sub-camadas), mais a figura
`etapa4_two_stream_espalhamento.png`.

FUNCOES PRINCIPAIS
--------------------
camada_two_stream(tau, omega0, g, D=2.0) -> (R, T)
    Refletancia e transmitancia de UMA camada homogenea iluminada
    difusamente por cima, sem fontes internas. Aceita tau/omega0/g
    escalares ou arrays (mesma forma). Casos conservativo (omega0=1) e
    nao-conservativo tratados separadamente (Eqs. 15.19-15.22 do curso).

combina_duas_camadas(Ra, Ta, Rb, Tb) -> (R_ab, T_ab)
    Metodo de adicao: combina camada de cima (a) com a de baixo (b) na
    R/T efetiva do conjunto (soma da serie infinita de reflexoes
    internas).

empilha_N_camadas(R_lista, T_lista) -> (R_total, T_total)
    Aplica combina_duas_camadas() sequencialmente a uma lista de camadas
    ordenadas do topo para a base.

delta_scaling(tau, omega0, g) -> (tau_l, omega0_l, g_l)
    Correcao delta-Eddington (f = g^2) para espalhamento com forte pico
    para frente, tipico de gotas de nuvem grandes.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1) CAMADA HOMOGENEA: REFLETANCIA E TRANSMITANCIA (Eqs. 15.19-15.22)
# ---------------------------------------------------------------------
def camada_two_stream(tau, omega0, g, D=2.0):
    """
    Refletancia (R) e transmitancia (T) de UMA camada homogenea,
    iluminada difusamente por cima, sem fontes internas (sem emissao
    termica, sem feixe direto tratado separadamente -- ver nota no
    cabecalho do arquivo).

    tau, omega0, g: escalares ou arrays (mesma forma).
    """
    tau = np.asarray(tau, dtype=float)
    omega0 = np.asarray(omega0, dtype=float)
    g = np.asarray(g, dtype=float)

    b = (1.0 - g) / 2.0  # fracao de retroespalhamento

    # --- Caso conservativo (omega0 == 1): Eq. 15.19-15.20 ---
    tau_escalado = (1.0 - g) * tau
    R_cons = tau_escalado / (2.0 + tau_escalado)
    T_cons = 2.0 / (2.0 + tau_escalado)

    # --- Caso nao-conservativo (omega0 < 1): Eq. 15.21-15.22 ---
    # formulas reescritas de forma numericamente estavel (dividindo por
    # e^{k tau} para evitar overflow quando k*tau for grande):
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.sqrt(np.maximum((1.0 - omega0) * D * ((1.0 - omega0) * D + 2.0 * omega0 * b), 0.0))
        gm = (1.0 - omega0) * D / np.where(k > 0, k, 1.0)
        gamma_mais = 1.0 + gm
        gamma_menos = 1.0 - gm

        e2 = np.exp(-2.0 * k * tau)
        denom = gamma_mais ** 2 - gamma_menos ** 2 * e2
        R_nc = gamma_mais * gamma_menos * (1.0 - e2) / denom
        T_nc = (gamma_mais ** 2 - gamma_menos ** 2) * np.exp(-k * tau) / denom

    conservativo = np.isclose(omega0, 1.0)
    R = np.where(conservativo, R_cons, R_nc)
    T = np.where(conservativo, T_cons, T_nc)
    return R, T


# ---------------------------------------------------------------------
# 1b) VALIDACAO CRUZADA: fechamento "two-stream approximation" do curso
#     (pag. 71-72 do material, coeficientes gamma com mu_1=1/sqrt(3),
#     Liou 1992; King e Harshvardhan 1986) -- formulacao INDEPENDENTE da
#     Secao 15 (D,b), usada aqui so para conferir que fechamentos
#     two-stream diferentes dao respostas fisicamente consistentes
#     (Meador e Weaver, 1980, mostram que varios fechamentos two-stream
#     partilham a mesma forma geral em termos de gamma_1, gamma_2).
# ---------------------------------------------------------------------
def camada_two_stream_curso(tau, omega0, g):
    """
    Refletancia/transmitancia usando os coeficientes gamma da
    "two-stream approximation" tal como apresentada no material do
    curso (pag. 72):
        mu_1 = 1/sqrt(3)
        gamma_1 = [1 - omega0*(1+g)/2] / mu_1
        gamma_2 = omega0*(1-g) / (2*mu_1)
    com a mesma forma fechada geral (kappa, R_inf) usada na Secao 15.
    """
    tau = np.asarray(tau, dtype=float)
    omega0 = np.asarray(omega0, dtype=float)
    g = np.asarray(g, dtype=float)

    mu1 = 1.0 / np.sqrt(3.0)
    gamma1 = (1.0 - omega0 * (1.0 + g) / 2.0) / mu1
    gamma2 = omega0 * (1.0 - g) / (2.0 * mu1)

    kappa = np.sqrt(np.maximum(gamma1 ** 2 - gamma2 ** 2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        Rinf = gamma2 / np.where(gamma1 + kappa > 0, gamma1 + kappa, 1.0)
        e2 = np.exp(-2.0 * kappa * tau)
        R = Rinf * (1.0 - e2) / (1.0 - Rinf ** 2 * e2)
        T = (1.0 - Rinf ** 2) * np.exp(-kappa * tau) / (1.0 - Rinf ** 2 * e2)

    # Caso conservativo (kappa->0): mesmo limite linear-em-tau da Secao 15
    conservativo = np.isclose(omega0, 1.0)
    tau_e = (1.0 - g) * tau
    R_cons = tau_e / (2.0 + tau_e)
    T_cons = 2.0 / (2.0 + tau_e)
    R = np.where(conservativo, R_cons, R)
    T = np.where(conservativo, T_cons, T)
    return R, T


# ---------------------------------------------------------------------
# 2) METODO DE ADICAO (Eq. 15.30-15.32) -- combina camadas empilhadas
# ---------------------------------------------------------------------
def combina_duas_camadas(Ra, Ta, Rb, Tb):
    """
    Combina duas camadas empilhadas (a = camada de cima, b = camada de
    baixo) na refletancia/transmitancia efetiva do conjunto, somando a
    serie infinita de reflexoes internas entre elas (adding method).
    """
    denom = 1.0 - Ra * Rb
    R_ab = Ra + (Ta ** 2) * Rb / denom
    T_ab = Ta * Tb / denom
    return R_ab, T_ab


def empilha_N_camadas(R_lista, T_lista):
    """R_lista/T_lista ordenadas do topo para a base."""
    R_tot, T_tot = R_lista[0], T_lista[0]
    for Ri, Ti in zip(R_lista[1:], T_lista[1:]):
        R_tot, T_tot = combina_duas_camadas(R_tot, T_tot, Ri, Ti)
    return R_tot, T_tot


# ---------------------------------------------------------------------
# 3) DELTA-SCALING (Secao 15.4c) -- corrige o pico de espalhamento p/ frente
# ---------------------------------------------------------------------
def delta_scaling(tau, omega0, g):
    f = g ** 2  # truncamento do 2o momento (f = chi/5 = g^2, cf. material do curso)
    g_l = (g - f) / (1.0 - f)
    tau_l = (1.0 - omega0 * f) * tau
    omega0_l = (1.0 - f) * omega0 / (1.0 - omega0 * f)
    return tau_l, omega0_l, g_l


if __name__ == "__main__":
    # ===================================================================
    # VALIDACAO 1: caso conservativo -- R + T deve ser exatamente 1
    # ===================================================================
    taus_teste = np.array([0.5, 1.0, 2.0, 5.0, 20.0])
    R_c, T_c = camada_two_stream(taus_teste, omega0=1.0, g=0.85)
    print("== Validacao 1: espalhamento conservativo (omega0=1) ==")
    print(f"  R+T (deve ser 1.0):  {R_c + T_c}")
    print(f"  Max |R+T - 1| = {np.max(np.abs(R_c + T_c - 1.0)):.2e}\n")

    # ===================================================================
    # VALIDACAO 2: caso nao-conservativo -- R+T < 1 (ha absorcao) e
    # limites fisicos (tau->0: T->1,R->0; tau->inf: T->0)
    # ===================================================================
    tau_pequeno = np.array([1e-4])
    tau_grande = np.array([50.0])
    R0, T0 = camada_two_stream(tau_pequeno, omega0=0.9, g=0.8)
    Rinf, Tinf = camada_two_stream(tau_grande, omega0=0.9, g=0.8)
    print("== Validacao 2: limites fisicos (omega0=0.9, g=0.8) ==")
    print(f"  tau->0:   R={R0[0]:.4f}  T={T0[0]:.4f}  (esperado: R~0, T~1)")
    print(f"  tau->inf: R={Rinf[0]:.4f}  T={Tinf[0]:.4f}  (esperado: T~0)\n")

    # ===================================================================
    # VALIDACAO 2b: fechamento "Secao 15" (D,b) x fechamento "two-stream
    # approximation" do curso (pag. 71-72, mu_1=1/sqrt(3)) -- devem
    # concordar razoavelmente (ambos sao aproximacoes validas da mesma
    # equacao de transferencia radiativa, cf. Meador e Weaver, 1980)
    # ===================================================================
    print("== Validacao 2b: fechamento da Secao 15 (D,b) vs 'two-stream approximation' "
          "do curso (pag. 71-72) ==")
    for omega0_t, g_t, tau_t in [(0.999, 0.85, 5.0), (0.95, 0.7, 2.0), (0.5, 0.3, 1.0)]:
        R1, T1 = camada_two_stream(tau_t, omega0_t, g_t)
        R2, T2 = camada_two_stream_curso(tau_t, omega0_t, g_t)
        print(f"  omega0={omega0_t}, g={g_t}, tau={tau_t}:  "
              f"Secao15 R={float(R1):.4f} T={float(T1):.4f}   |   "
              f"curso(pag.72) R={float(R2):.4f} T={float(T2):.4f}   "
              f"(dif. R={abs(float(R1)-float(R2)):.4f})")
    print()

    # ===================================================================
    # VALIDACAO 3: metodo de adicao deve reproduzir a formula de camada
    # unica quando subdividimos a MESMA camada em N pedacos identicos
    # ===================================================================
    tau_total, omega0_teste, g_teste = 8.0, 0.95, 0.85
    R_direto, T_direto = camada_two_stream(tau_total, omega0_teste, g_teste)

    for n_sub in [2, 5, 20]:
        tau_sub = tau_total / n_sub
        R_sub, T_sub = camada_two_stream(tau_sub, omega0_teste, g_teste)
        R_lista = [R_sub] * n_sub
        T_lista = [T_sub] * n_sub
        R_emp, T_emp = empilha_N_camadas(R_lista, T_lista)
        print(f"  N_sub={n_sub:2d}: R_empilhado={R_emp:.6f}  T_empilhado={T_emp:.6f}  "
              f"(direto: R={R_direto:.6f} T={T_direto:.6f}, "
              f"dif={abs(R_emp - R_direto):.2e})")
    print()

    # ===================================================================
    # APLICACAO: albedo do sistema NUVEM + SUPERFICIE vs espessura optica
    # ===================================================================
    albedo_superficie = 0.1  # ex.: oceano
    # superficie tratada como uma "camada" com R=albedo, T=0 (opaca)
    R_sfc = np.full_like(np.array([1.0]), albedo_superficie)[0]
    T_sfc = 0.0

    taus_nuvem = np.logspace(-1, 2, 60)  # 0.1 a 100

    # Tres tamanhos de gota tipicos (g diferente) e omega0 quase conservativo
    casos = [
        dict(g=0.70, omega0=0.9995, nome="gotas pequenas (g=0.70)"),
        dict(g=0.85, omega0=0.9997, nome="gotas tipicas (g=0.85)"),
        dict(g=0.95, omega0=0.9999, nome="gotas grandes (g=0.95)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Albedo do sistema vs tau_nuvem, para diferentes g
    ax = axes[0, 0]
    for caso in casos:
        R_nuvem, T_nuvem = camada_two_stream(taus_nuvem, caso["omega0"], caso["g"])
        R_sistema = []
        for Rn, Tn in zip(R_nuvem, T_nuvem):
            R_tot, _ = combina_duas_camadas(Rn, Tn, R_sfc, T_sfc)
            R_sistema.append(R_tot)
        ax.plot(taus_nuvem, R_sistema, label=caso["nome"])
    ax.axhline(albedo_superficie, color="gray", ls=":", lw=1, label="ceu limpo (so superficie)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Espessura optica da nuvem, $\tau$")
    ax.set_ylabel("Albedo do sistema nuvem+superficie")
    ax.set_title("Albedo vs espessura optica e tamanho de particula\n(cf. Fig. 15.4 do material do curso)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Efeito do delta-scaling numa nuvem de forte espalhamento p/ frente
    ax = axes[0, 1]
    g_alto, omega0_alto = 0.90, 0.9998
    R_sem_delta, T_sem_delta = camada_two_stream(taus_nuvem, omega0_alto, g_alto)
    tau_l, omega0_l, g_l = delta_scaling(taus_nuvem, omega0_alto, g_alto)
    R_com_delta, T_com_delta = camada_two_stream(tau_l, omega0_l, g_l)
    ax.plot(taus_nuvem, R_sem_delta, label="sem delta-scaling")
    ax.plot(taus_nuvem, R_com_delta, "--", label="com delta-scaling")
    ax.set_xscale("log")
    ax.set_xlabel(r"Espessura optica, $\tau$")
    ax.set_ylabel("Refletancia da camada")
    ax.set_title(f"Efeito do delta-scaling (g={g_alto}, forte espalhamento p/ frente)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Sensibilidade ao albedo de superficie (oceano vs neve)
    ax = axes[1, 0]
    for alb_sfc, nome in [(0.06, "oceano (0.06)"), (0.2, "solo/vegetacao (0.2)"), (0.8, "neve/gelo (0.8)")]:
        R_nuvem, T_nuvem = camada_two_stream(taus_nuvem, 0.9997, 0.85)
        R_sistema = [combina_duas_camadas(Rn, Tn, alb_sfc, 0.0)[0] for Rn, Tn in zip(R_nuvem, T_nuvem)]
        ax.plot(taus_nuvem, R_sistema, label=nome)
    ax.set_xscale("log")
    ax.set_xlabel(r"Espessura optica da nuvem, $\tau$")
    ax.set_ylabel("Albedo do sistema")
    ax.set_title("Nuvens finas: a superficie ainda importa\nNuvens espessas: a superficie deixa de importar")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Transmitancia (quanto de SW chega na superficie) vs tau_nuvem
    ax = axes[1, 1]
    for caso in casos:
        R_nuvem, T_nuvem = camada_two_stream(taus_nuvem, caso["omega0"], caso["g"])
        ax.plot(taus_nuvem, T_nuvem, label=caso["nome"])
    ax.set_xscale("log")
    ax.set_xlabel(r"Espessura optica da nuvem, $\tau$")
    ax.set_ylabel("Transmitancia da nuvem")
    ax.set_title("Fracao de SW que atravessa a nuvem\n(o que sobra para aquecer a superficie)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("./etapa4_two_stream_espalhamento.png", dpi=150)
    print("Figura salva em etapa4_two_stream_espalhamento.png")
