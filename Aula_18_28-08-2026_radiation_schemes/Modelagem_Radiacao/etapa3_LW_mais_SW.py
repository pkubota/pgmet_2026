"""
ETAPA 3 - Introduzindo a onda curta (SW) NA ATMOSFERA
=======================================================

Ate a Etapa 2, toda a radiacao solar era absorvida so na superficie
(atmosfera "transparente" a SW). Isso produz um perfil de equilibrio que
so esfria com a altura -- sem a inversao de temperatura da estratosfera
real, causada pela absorcao de UV/visivel pelo OZONIO (e, em menor grau,
H2O e CO2 tambem absorvem SW).

O que fazemos aqui (ainda sem espalhamento -- Etapa 4):
  1. Cada camada agora tambem pode ABSORVER parte do feixe solar direto
     (lei de Beer, sem reemissao -- correto para SW, ver Lecture 21 do
     material do curso: "No emission in the solar... only heating due to
     absorption").
  2. Construimos perfis simplificados (mas fisicamente motivados) de
     absorcao de O3 (concentrado na estratosfera, pico ~10 hPa), H2O
     (concentrado perto da superficie) e CO2 (bem misturado por massa).
  3. Generalizamos a solucao fechada de equilibrio radiativo da Etapa 1/2
     para incluir esses "termos de fonte" Q_i por camada -- exatamente
     o papel dos termos Q+/Q- na equacao geral de two-stream do material
     do curso (Eq. 15.15/15.25), aqui na versao sem espalhamento.
  4. Comparamos o novo perfil de equilibrio (LW+SW) com o da Etapa 2
     (so LW) para ver a estratosfera aquecer.

COMO EXECUTAR
--------------
    python3 etapa3_LW_mais_SW.py

Parametros editaveis no `if __name__ == "__main__":`: N, ps/ptop (grade
log-espacada, necessaria para resolver a camada de ozonio), tau_total_lw.
As fracoes de absorcao solar por gas (frac_o3, frac_h2o, frac_co2) e as
formas verticais estao nos argumentos-padrao de `perfis_absorcao_sw`.

SAIDA: prints comparando o equilibrio so-LW (Etapa 2) com o LW+SW, e a
figura `etapa3_LW_mais_SW.png` (4 paineis).

FUNCOES PRINCIPAIS
--------------------
perfis_absorcao_sw(p_niveis, S_abs, frac_o3=0.02, frac_h2o=0.15,
                    frac_co2=0.01, p_pico_o3=10.0, sigma_lnp_o3=0.7,
                    H_h2o=2.5, H_atm=7.4) -> (Q_o3, Q_h2o, Q_co2, Q_total)
    Energia solar absorvida (W/m2) por camada e por gas -- formas
    idealizadas (gaussiana em ln(p) para O3, exponencial em z para H2O,
    proporcional a massa para CO2).

modelo_N_camadas_equilibrio_com_fontes(S_surf, Q_camadas, eps)
    -> (T_camadas, Ts, F_dn, S)
    Generaliza a Etapa 1/2: o fluxo liquido de LW deixa de ser constante
    com a altura, crescendo a cada camada que absorve SW (S_j = S_{j-1} +
    Q_{j-1}). Reduz-se a etapa2 se Q_camadas for tudo zero.

relaxa_para_equilibrio_com_sw(T0, Ts0, eps, p_niveis, Q_camadas, S_surf,
                               dt_s=21600.0, n_passos=1600)
    Marcha no tempo equivalente, agora somando a taxa de aquecimento SW
    (fixa, nao depende de T) a taxa LW (recalculada a cada passo).
"""

import numpy as np
import matplotlib.pyplot as plt

SIGMA = 5.670374419e-8
G = 9.81
CP = 1004.0


def fluxo_solar_absorvido(S0=1361.0, albedo=0.3):
    """
    NOTA: estes sao os valores de balanco de energia GLOBAL-MEDIO (o
    fator 1/4 ja e a media geometrica dia/noite e latitude). O material
    do curso (Fig. 21.2/21.3, Lecture 21) usa parametros diferentes para
    calculos INSTANTANEOS de taxa de aquecimento solar (S0=1360 W/m2,
    albedo de superficie=0.15, cos(zenith)=0.5, 12h de sol) -- nao
    diretamente comparaveis ao balanco global-medio usado aqui para achar
    o equilibrio radiativo da coluna.
    """
    return S0 * (1.0 - albedo) / 4.0


# ---------------------------------------------------------------------
# GRADE VERTICAL (mesma logica da Etapa 2, agora indo ate 1 hPa para
# capturar a camada de ozonio)
# ---------------------------------------------------------------------
def grade_pressao(N, ps=1013.25, ptop=1.0):
    """Grade log-espacada em pressao -- da mais resolucao perto do topo
    (estratosfera), onde a camada de ozonio precisa ser bem resolvida."""
    return np.exp(np.linspace(np.log(ps), np.log(ptop), N + 1))


DIFUSIVIDADE = 1.66
"""Fator de difusividade (Elsasser 1942, Eq. 4.9 do material do curso);
mesmo valor usado operacionalmente pelo RRTMG_LW (pag. 106 do material)."""


def espessura_optica_lw(p_niveis, tau_total, D=DIFUSIVIDADE):
    ps = p_niveis[0]
    dp = p_niveis[:-1] - p_niveis[1:]
    dtau = tau_total * dp / ps
    eps = 1.0 - np.exp(-D * dtau)
    return eps, dtau


def calcula_fluxos_lw(T_camadas, Ts, eps):
    eps = np.asarray(eps, dtype=float)
    N = len(eps)
    t = 1.0 - eps
    F_up = np.zeros(N + 1)
    F_up[0] = SIGMA * Ts ** 4
    for i in range(N):
        F_up[i + 1] = t[i] * F_up[i] + eps[i] * SIGMA * T_camadas[i] ** 4
    F_dn = np.zeros(N + 1)
    for i in range(N - 1, -1, -1):
        F_dn[i] = t[i] * F_dn[i + 1] + eps[i] * SIGMA * T_camadas[i] ** 4
    return F_up, F_dn


def taxa_aquecimento_lw(F_up, F_dn, p_niveis, cp=CP, g=G):
    F_net = F_up - F_dn
    dp_pa = (p_niveis[:-1] - p_niveis[1:]) * 100.0
    conv = F_net[:-1] - F_net[1:]
    return (g / (cp * dp_pa)) * conv * 86400.0  # K/dia


# ---------------------------------------------------------------------
# PERFIS DE ABSORCAO SOLAR (O3, H2O, CO2) -- formas idealizadas
# ---------------------------------------------------------------------
def perfis_absorcao_sw(p_niveis, S_abs,
                        frac_o3=0.02, frac_h2o=0.15, frac_co2=0.01,
                        p_pico_o3=10.0, sigma_lnp_o3=0.7,
                        H_h2o=2.5, H_atm=7.4):
    """
    Retorna Q_o3, Q_h2o, Q_co2, Q_total: energia SW absorvida (W/m2) em
    cada camada, com o total de cada gas somando frac_gas * S_abs.

    - Ozonio: forma gaussiana em ln(p), pico em p_pico_o3 (hPa) --
      mimetiza a camada de ozonio estratosferica.
    - Vapor d'agua: decai exponencialmente com a altura (H_h2o ~ poucos km),
      concentrado na baixa troposfera.
    - CO2: bem misturado por massa (mesma logica do perfil de eps_LW).
    """
    p_centros = 0.5 * (p_niveis[:-1] + p_niveis[1:])
    ps = p_niveis[0]
    dp = p_niveis[:-1] - p_niveis[1:]  # espessura em pressao de cada camada
    z_centros = -H_atm * np.log(p_centros / ps)  # altitude aproximada (km)

    # --- Ozonio: gaussiana em ln(p) ---
    forma_o3 = np.exp(-0.5 * ((np.log(p_centros) - np.log(p_pico_o3)) / sigma_lnp_o3) ** 2)
    forma_o3 /= forma_o3.sum()
    Q_o3 = frac_o3 * S_abs * forma_o3

    # --- Vapor d'agua: decaimento exponencial com a altura ---
    forma_h2o = np.exp(-z_centros / H_h2o)
    forma_h2o /= forma_h2o.sum()
    Q_h2o = frac_h2o * S_abs * forma_h2o

    # --- CO2: bem misturado por massa (proporcional a dp) ---
    forma_co2 = dp / dp.sum()
    Q_co2 = frac_co2 * S_abs * forma_co2

    Q_total = Q_o3 + Q_h2o + Q_co2
    return Q_o3, Q_h2o, Q_co2, Q_total


# ---------------------------------------------------------------------
# SOLUCAO FECHADA DE EQUILIBRIO RADIATIVO, AGORA COM FONTES SW (Q_i)
# ---------------------------------------------------------------------
def modelo_N_camadas_equilibrio_com_fontes(S_surf, Q_camadas, eps):
    """
    Generalizacao da formula da Etapa 1/2: em vez de assumir que o fluxo
    liquido de LW e constante com a altura, agora ele CRESCE a cada
    camada que absorve SW (Q_i), pois em equilibrio a divergencia de LW
    tem que compensar a fonte solar local:
        F_net_LW[nivel j+1] = F_net_LW[nivel j] + Q_j

    eps[i] / Q_camadas[i]: camada i=0 e a mais proxima da superficie.
    S_surf: energia solar absorvida diretamente pela superficie
            (S_surf = S_abs - soma(Q_camadas)).
    """
    eps = np.asarray(eps, dtype=float)
    Q = np.asarray(Q_camadas, dtype=float)
    N = len(eps)
    t = 1.0 - eps

    # fluxo liquido de LW "alvo" em cada nivel (cresce com a altura ao
    # atravessar camadas que absorvem SW)
    S = np.zeros(N + 1)
    S[0] = S_surf
    for i in range(N):
        S[i + 1] = S[i] + Q[i]

    F_dn = np.zeros(N + 1)
    for j in range(N, 0, -1):
        num = S[j] - t[j - 1] * S[j - 1]
        F_dn[j - 1] = F_dn[j] + num / (1.0 + t[j - 1])

    x_camadas = np.zeros(N)
    for j in range(N, 0, -1):
        num = S[j] - t[j - 1] * S[j - 1]
        x_camadas[j - 1] = F_dn[j] + num / ((1.0 + t[j - 1]) * eps[j - 1])
    T_camadas = (x_camadas / SIGMA) ** 0.25

    Ts4 = S[0] + F_dn[0]
    Ts = (Ts4 / SIGMA) ** 0.25

    return T_camadas, Ts, F_dn, S


# ---------------------------------------------------------------------
# MARCHA NO TEMPO COM LW + SW (validacao numerica da formula fechada)
# ---------------------------------------------------------------------
def relaxa_para_equilibrio_com_sw(T0, Ts0, eps, p_niveis, Q_camadas, S_surf,
                                   dt_s=6 * 3600.0, n_passos=1600):
    T = T0.copy()
    Ts = Ts0
    dp_pa = (p_niveis[:-1] - p_niveis[1:]) * 100.0
    dTdt_sw = (G / (CP * dp_pa)) * Q_camadas * 86400.0  # K/dia, fixo no tempo

    hist_T = [T.copy()]
    hist_Ts = [Ts]
    for _ in range(n_passos):
        F_up, F_dn = calcula_fluxos_lw(T, Ts, eps)
        dTdt_lw = taxa_aquecimento_lw(F_up, F_dn, p_niveis)
        T = T + (dTdt_lw + dTdt_sw) * (dt_s / 86400.0)
        Ts = ((S_surf + F_dn[0]) / SIGMA) ** 0.25
        hist_T.append(T.copy())
        hist_Ts.append(Ts)

    return T, Ts, np.array(hist_T), np.array(hist_Ts)


if __name__ == "__main__":
    N = 40
    ps, ptop = 1013.25, 1.0
    tau_total_lw = 4.0
    S_abs = fluxo_solar_absorvido()
    Te = (S_abs / SIGMA) ** 0.25

    p_niveis = grade_pressao(N, ps, ptop)
    p_centros = 0.5 * (p_niveis[:-1] + p_niveis[1:])
    eps, dtau = espessura_optica_lw(p_niveis, tau_total_lw)

    Q_o3, Q_h2o, Q_co2, Q_total = perfis_absorcao_sw(p_niveis, S_abs)
    S_surf = S_abs - Q_total.sum()

    print(f"S_abs = {S_abs:.2f} W/m2   Te = {Te:.2f} K")
    print(f"SW absorvido na atmosfera: O3={Q_o3.sum():.2f}  H2O={Q_h2o.sum():.2f}  "
          f"CO2={Q_co2.sum():.2f}  TOTAL={Q_total.sum():.2f} W/m2")
    print(f"SW absorvido na superficie: S_surf = {S_surf:.2f} W/m2\n")

    # --- Taxas de aquecimento SW instantaneas (nao dependem da temperatura) ---
    dp_pa = (p_niveis[:-1] - p_niveis[1:]) * 100.0
    dTdt_o3 = (G / (CP * dp_pa)) * Q_o3 * 86400.0
    dTdt_h2o = (G / (CP * dp_pa)) * Q_h2o * 86400.0
    dTdt_co2 = (G / (CP * dp_pa)) * Q_co2 * 86400.0
    dTdt_sw_total = dTdt_o3 + dTdt_h2o + dTdt_co2
    print(f"Pico do aquecimento SW por ozonio: {dTdt_o3.max():.2f} K/dia "
          f"em p={p_centros[np.argmax(dTdt_o3)]:.1f} hPa\n")

    # --- Equilibrio radiativo LW-only (Etapa 2, Q=0) para comparacao ---
    T_eq_lw, Ts_eq_lw, _, _ = modelo_N_camadas_equilibrio_com_fontes(
        S_abs, np.zeros(N), eps)

    # --- Equilibrio radiativo LW+SW (Etapa 3) -- solucao fechada --------
    T_eq_swlw, Ts_eq_swlw, F_dn_eq, S_eq = modelo_N_camadas_equilibrio_com_fontes(
        S_surf, Q_total, eps)

    print("== Comparacao dos equilibrios ==")
    print(f"  So LW  (Etapa 2): T_superficie = {Ts_eq_lw:.2f} K")
    print(f"  LW+SW  (Etapa 3): T_superficie = {Ts_eq_swlw:.2f} K")
    print(f"  Temperatura minima do perfil LW+SW: {T_eq_swlw.min():.2f} K "
          f"em p={p_centros[np.argmin(T_eq_swlw)]:.1f} hPa  "
          f"(marca a 'tropopausa' deste modelo simples)\n")

    # --- Validacao: marcha no tempo deve bater com a formula fechada ----
    T0 = np.full(N, Te)
    Ts0 = Te + 10.0
    T_final, Ts_final, hist_T, hist_Ts = relaxa_para_equilibrio_com_sw(
        T0, Ts0, eps, p_niveis, Q_total, S_surf, n_passos=1600)
    erro = np.max(np.abs(T_final - T_eq_swlw))
    print(f"Validacao (marcha no tempo x formula fechada): "
          f"maior diferenca = {erro:.4f} K, dif. Ts = {abs(Ts_final-Ts_eq_swlw):.4f} K\n")

    # ---------------------------- FIGURAS ------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Taxas de aquecimento SW por gas
    ax = axes[0, 0]
    ax.plot(dTdt_o3, p_centros, label="O3 (ozonio)")
    ax.plot(dTdt_h2o, p_centros, label="H2O (vapor d'agua)")
    ax.plot(dTdt_co2, p_centros, label="CO2")
    ax.plot(dTdt_sw_total, p_centros, "k--", label="Total SW")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Taxa de aquecimento SW (K/dia)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("Aquecimento solar por gas\n(perfis idealizados, cf. Fig. 21.1 do curso)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (b) Perfil de equilibrio: so-LW x LW+SW
    ax = axes[0, 1]
    ax.plot(T_eq_lw, p_centros, "--", label="So LW (Etapa 2)")
    ax.plot(T_eq_swlw, p_centros, "-o", ms=2, label="LW + SW (Etapa 3)")
    ax.axvline(Te, color="gray", ls=":", lw=1, label="T_efetiva (Te)")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Temperatura (K)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("Equilibrio radiativo: efeito da absorcao de O3\n(aparece a 'estratosfera' aquecida)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (c) Evolucao temporal ate o equilibrio (algumas camadas)
    ax = axes[1, 0]
    dias = np.arange(hist_T.shape[0]) * (6 * 3600.0) / 86400.0
    idx_o3 = int(np.argmax(Q_o3))
    for idx, nome, cor in [(0, "perto da superficie", "tab:blue"),
                            (idx_o3, "pico do ozonio", "tab:green"),
                            (N - 1, "topo do dominio", "tab:purple")]:
        ax.plot(dias, hist_T[:, idx], color=cor,
                label=f"{nome} (p~{p_centros[idx]:.0f} hPa)")
    ax.plot(dias, hist_Ts, color="tab:red", label="superficie")
    ax.set_xlabel("Tempo (dias)")
    ax.set_ylabel("Temperatura (K)")
    ax.set_title("Marcha no tempo ate o equilibrio (LW+SW)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (d) Fluxo liquido de LW em funcao da altura (mostra que nao e mais constante)
    ax = axes[1, 1]
    ax.plot(S_eq, p_niveis, "-o", ms=2)
    ax.axvline(S_abs, color="gray", ls=":", lw=1, label="S_abs (=OLR no topo)")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Fluxo liquido de LW (W/m2)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("Fluxo liquido de LW cresce com a altura\n(compensando a absorcao de SW em cada camada)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("./etapa3_LW_mais_SW.png", dpi=150)
    print("Figura salva em etapa3_LW_mais_SW.png")
