"""
ETAPA 5 - De cinza para MULTIPLAS BANDAS ESPECTRAIS (LW)
============================================================

Ate a Etapa 4, a atmosfera era "cinza": cada camada tinha uma unica
emissividade, igual para todos os comprimentos de onda. Isso e uma
simplificacao grosseira -- na realidade, a atmosfera e quase transparente
na "janela atmosferica" (8-12 um) e fortemente opaca nas bandas de
absorcao do H2O, CO2 e O3.

O que mudamos aqui:
  1. Dividimos o espectro LW em ~7 bandas largas, cada uma dominada por
     um ou dois absorvedores (H2O, CO2, O3), com coeficientes de absorcao
     em massa idealizados (nao vem de dados espectroscopicos reais --
     isso sera substituido por k-distribution/correlated-k na Etapa 6).
  2. Cada camada agora tem uma emissividade DIFERENTE por banda,
     calculada com a lei de Beer a partir do caminho optico de cada gas
     (Delta_tau = k_banda,gas * massa_gas_na_camada).
  3. A emissao de cada camada em cada banda usa a FRACAO da funcao de
     Planck que cai dentro daquela banda -- B_banda(T) = f_banda(T)*sigma*T^4
     -- em vez de sigma*T^4 inteiro. Isso e nao-linear em T (a fracao
     muda com a temperatura), entao a solucao fechada da Etapa 1-3 nao
     se aplica mais: resolvemos por marcha no tempo (como na Etapa 2).
  4. Validamos o codigo espectral fazendo ele reproduzir EXATAMENTE o
     resultado cinza da Etapa 2, quando forcamos a mesma emissividade em
     todas as bandas.

COMO EXECUTAR
--------------
    python3 etapa5_bandas_espectrais.py

Aviso: a primeira execucao recalcula a tabela de fracoes de Planck por
banda (arrays _T_TABELA/_FRAC_TABELA, calculados no carregamento do
modulo) -- leva alguns segundos. Os coeficientes de absorcao por banda
(K_H2O, K_CO2, K_O3, no topo do arquivo) sao os valores JA RECALIBRADOS
(ver discussao no Capitulo 4 do relatorio); os valores originais mais
fortes que causaram saturacao estao comentados ali ao lado.

SAIDA: prints com a validacao (bandas identicas = resultado cinza) e o
caso real por banda, mais a figura `etapa5_bandas_espectrais.png`.

FUNCOES PRINCIPAIS
--------------------
fracao_planck_banda(T, nu1, nu2, n=300) -> fracao (0-1)
    Fracao da emissao de corpo negro total que cai entre nu1 e nu2 (cm^-1)
    para a temperatura T, por integracao numerica da funcao de Planck.

B_banda(T, ib) -> W/m2
    Emissao de corpo negro na banda de indice `ib` (ver lista BANDAS no
    topo do arquivo), usando interpolacao na tabela pre-calculada.

espessura_optica_bandas(p_niveis, q_h2o, q_co2, q_o3) -> (eps, dtau)
    eps tem forma (N_BANDAS, N_camadas). Cada banda usa os coeficientes
    K_H2O[ib]/K_CO2[ib]/K_O3[ib] (m2/kg) aplicados ao caminho de massa de
    cada gas na camada.

fluxos_totais(T_camadas, Ts, eps) -> (F_up_tot, F_dn_tot, F_up_bandas, F_dn_bandas)
    Soma os fluxos de todas as N_BANDAS=7 bandas.

relaxa_equilibrio_espectral(T0, Ts0, eps, p_niveis, S_abs, dt_s=21600.0,
                             n_passos=1500) -> (T_final, Ts_final, hist_T, hist_Ts)
    Marcha no tempo multi-banda (nao ha formula fechada para o caso
    espectral, pois a fracao de Planck por banda depende nao-linearmente
    de T).
"""

import numpy as np
import matplotlib.pyplot as plt

SIGMA = 5.670374419e-8
G = 9.81
CP = 1004.0
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23


def fluxo_solar_absorvido(S0=1361.0, albedo=0.3):
    return S0 * (1.0 - albedo) / 4.0


# ---------------------------------------------------------------------
# BANDAS ESPECTRAIS (cm^-1) -- larguras idealizadas, cf. estrutura real
# ---------------------------------------------------------------------
BANDAS = [(1, 200), (200, 500), (500, 800), (800, 980),
          (980, 1100), (1100, 1500), (1500, 3000)]
NOMES_BANDAS = ["Far-IR\n(H2O rot.)", "H2O rot.", "CO2 15um",
                "Janela\natmosferica", "O3 9.6um", "H2O vib-rot", "Fraco"]
N_BANDAS = len(BANDAS)

# Coeficientes de absorcao em massa idealizados (m2/kg) por banda e gas.
# NOTA: uma primeira tentativa com coeficientes maiores (K_H2O de pico=30,
# K_CO2=5, K_O3=100) saturou quase todas as bandas (tau_coluna > 100),
# levando a um efeito estufa exagerado (Ts~362K!). Isso e um efeito
# fisico real de tratar a banda como "cinza": sem a estrutura de linhas
# espectrais real (nucleos saturados + asas fracas que ainda deixam
# radiacao escapar), o Beer's law simples SUPERESTIMA a opacidade
# efetiva de uma banda larga -- exatamente o problema que motiva o
# k-distribution/correlated-k na Etapa 6. Os valores abaixo foram
# reduzidos para dar espessuras oticas mais plausiveis (mantendo a
# ideia pedagogica: janela transparente, bandas de H2O/CO2/O3 opacas).
K_H2O = np.array([3.0, 1.0, 0.35, 0.02, 0.05, 0.9, 0.05])
K_CO2 = np.array([0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.05])
K_O3 = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0])


# ---------------------------------------------------------------------
# FUNCAO DE PLANCK POR BANDA (fracao da emissao de corpo negro na banda)
# ---------------------------------------------------------------------
def planck_nu(nu_cm, T):
    """Radiancia espectral de corpo negro (W/m2/sr/m^-1), nu em cm^-1."""
    nu_m = nu_cm * 100.0
    x = H_PLANCK * C_LIGHT * nu_m / (K_BOLTZ * T)
    x = np.clip(x, 1e-10, 700.0)
    return 2.0 * H_PLANCK * C_LIGHT ** 2 * nu_m ** 3 / (np.exp(x) - 1.0)


def fracao_planck_banda(T, nu1, nu2, n=300):
    nu = np.linspace(nu1, nu2, n)
    B = planck_nu(nu, T)
    integral = np.trapezoid(B, nu * 100.0)  # integra em m^-1
    return np.pi * integral / (SIGMA * T ** 4)


# Tabela pre-computada (T de 150 a 340 K) para interpolacao rapida durante
# a marcha no tempo -- evita reintegrar a funcao de Planck a cada passo.
_T_TABELA = np.arange(150.0, 341.0, 1.0)
_FRAC_TABELA = np.zeros((N_BANDAS, len(_T_TABELA)))
for _ib, (_nu1, _nu2) in enumerate(BANDAS):
    for _it, _T in enumerate(_T_TABELA):
        _FRAC_TABELA[_ib, _it] = fracao_planck_banda(_T, _nu1, _nu2)
_FRAC_TABELA /= _FRAC_TABELA.sum(axis=0, keepdims=True)  # normaliza p/ somar 1


def B_banda(T, ib):
    """Emissao de corpo negro na banda ib, para temperatura(s) T (K)."""
    frac = np.interp(T, _T_TABELA, _FRAC_TABELA[ib])
    return frac * SIGMA * T ** 4


# ---------------------------------------------------------------------
# GRADE VERTICAL E PERFIS DE GASES (mesma logica da Etapa 3)
# ---------------------------------------------------------------------
def grade_pressao(N, ps=1013.25, ptop=1.0):
    return np.exp(np.linspace(np.log(ps), np.log(ptop), N + 1))


def perfis_gases(p_niveis, H_atm=7.4):
    p_centros = 0.5 * (p_niveis[:-1] + p_niveis[1:])
    ps = p_niveis[0]
    z = -H_atm * np.log(p_centros / ps)
    q_h2o = 0.010 * np.exp(-z / 2.5)               # kg/kg
    q_co2 = np.full_like(p_centros, 6.08e-4)         # kg/kg (~400 ppmv)
    q_o3 = 8e-6 * np.exp(-0.5 * ((np.log(p_centros) - np.log(10.0)) / 0.7) ** 2)  # kg/kg
    return q_h2o, q_co2, q_o3


DIFUSIVIDADE = 1.66
"""Fator de difusividade (Elsasser 1942, Eq. 4.9 do material do curso);
mesmo valor usado operacionalmente pelo RRTMG_LW (pag. 106 do material)."""


def espessura_optica_bandas(p_niveis, q_h2o, q_co2, q_o3, D=DIFUSIVIDADE):
    dp_pa = (p_niveis[:-1] - p_niveis[1:]) * 100.0
    m_h2o, m_co2, m_o3 = q_h2o * dp_pa / G, q_co2 * dp_pa / G, q_o3 * dp_pa / G
    N = len(q_h2o)
    dtau = np.zeros((N_BANDAS, N))
    for ib in range(N_BANDAS):
        dtau[ib] = K_H2O[ib] * m_h2o + K_CO2[ib] * m_co2 + K_O3[ib] * m_o3
    eps = 1.0 - np.exp(-D * dtau)
    return eps, dtau


# ---------------------------------------------------------------------
# FLUXOS POR BANDA E TOTAL
# ---------------------------------------------------------------------
def calcula_fluxos_banda(T_camadas, Ts, eps_banda, ib):
    N = len(eps_banda)
    t = 1.0 - eps_banda
    F_up = np.zeros(N + 1)
    F_up[0] = B_banda(Ts, ib)
    for i in range(N):
        F_up[i + 1] = t[i] * F_up[i] + eps_banda[i] * B_banda(T_camadas[i], ib)
    F_dn = np.zeros(N + 1)
    for i in range(N - 1, -1, -1):
        F_dn[i] = t[i] * F_dn[i + 1] + eps_banda[i] * B_banda(T_camadas[i], ib)
    return F_up, F_dn


def fluxos_totais(T_camadas, Ts, eps):
    N = eps.shape[1]
    F_up_tot, F_dn_tot = np.zeros(N + 1), np.zeros(N + 1)
    F_up_bandas, F_dn_bandas = [], []
    for ib in range(N_BANDAS):
        Fu, Fd = calcula_fluxos_banda(T_camadas, Ts, eps[ib], ib)
        F_up_tot += Fu
        F_dn_tot += Fd
        F_up_bandas.append(Fu)
        F_dn_bandas.append(Fd)
    return F_up_tot, F_dn_tot, F_up_bandas, F_dn_bandas


def taxa_aquecimento(F_up, F_dn, p_niveis):
    F_net = F_up - F_dn
    dp_pa = (p_niveis[:-1] - p_niveis[1:]) * 100.0
    conv = F_net[:-1] - F_net[1:]
    return (G / (CP * dp_pa)) * conv * 86400.0


def relaxa_equilibrio_espectral(T0, Ts0, eps, p_niveis, S_abs,
                                 dt_s=6 * 3600.0, n_passos=1500):
    T, Ts = T0.copy(), Ts0
    hist_T, hist_Ts = [T.copy()], [Ts]
    for _ in range(n_passos):
        F_up, F_dn, _, _ = fluxos_totais(T, Ts, eps)
        dTdt = taxa_aquecimento(F_up, F_dn, p_niveis)
        T = T + dTdt * (dt_s / 86400.0)
        Ts = ((S_abs + F_dn[0]) / SIGMA) ** 0.25
        hist_T.append(T.copy())
        hist_Ts.append(Ts)
    return T, Ts, np.array(hist_T), np.array(hist_Ts)


if __name__ == "__main__":
    N = 30
    S_abs = fluxo_solar_absorvido()
    Te = (S_abs / SIGMA) ** 0.25
    p_niveis = grade_pressao(N)
    p_centros = 0.5 * (p_niveis[:-1] + p_niveis[1:])

    # =================================================================
    # VALIDACAO: replicar a mesma emissividade em TODAS as bandas deve
    # reproduzir exatamente o resultado cinza da Etapa 2.
    # =================================================================
    tau_total_gray = 4.0
    dp = p_niveis[:-1] - p_niveis[1:]
    dtau_gray = tau_total_gray * dp / p_niveis[0]
    eps_gray = 1.0 - np.exp(-dtau_gray)
    eps_replicado = np.tile(eps_gray, (N_BANDAS, 1))

    T0 = np.full(N, Te)
    Ts0 = Te + 10.0
    T_val, Ts_val, _, _ = relaxa_equilibrio_espectral(
        T0, Ts0, eps_replicado, p_niveis, S_abs, n_passos=1500)

    # formula fechada cinza (Etapa 1/2) para comparar:
    def equilibrio_cinza(S_abs, eps):
        Np = len(eps)
        t = 1 - eps
        F_dn = np.zeros(Np + 1)
        for j in range(Np, 0, -1):
            F_dn[j - 1] = F_dn[j] + S_abs * eps[j - 1] / (1 + t[j - 1])
        x = np.zeros(Np)
        for j in range(Np, 0, -1):
            x[j - 1] = F_dn[j] + S_abs / (1 + t[j - 1])
        Ts = ((S_abs + F_dn[0]) / SIGMA) ** 0.25
        return (x / SIGMA) ** 0.25, Ts

    T_gray, Ts_gray = equilibrio_cinza(S_abs, eps_gray)
    print("== Validacao: bandas identicas devem reproduzir o resultado cinza ==")
    print(f"  T_superficie espectral (bandas replicadas) = {Ts_val:.3f} K")
    print(f"  T_superficie cinza (Etapa 1/2, formula fechada) = {Ts_gray:.3f} K")
    print(f"  Maior diferenca entre camadas: {np.max(np.abs(T_val - T_gray)):.4f} K\n")

    # =================================================================
    # CASO REAL: emissividade diferente por banda (H2O, CO2, O3)
    # =================================================================
    q_h2o, q_co2, q_o3 = perfis_gases(p_niveis)
    eps_bandas, dtau_bandas = espessura_optica_bandas(p_niveis, q_h2o, q_co2, q_o3)

    print("== Espessura optica total (coluna) por banda ==")
    for ib in range(N_BANDAS):
        print(f"  {NOMES_BANDAS[ib].replace(chr(10),' '):20s}: "
              f"tau_coluna = {dtau_bandas[ib].sum():8.3f}")
    print()

    T_final, Ts_final, hist_T, hist_Ts = relaxa_equilibrio_espectral(
        T0, Ts0, eps_bandas, p_niveis, S_abs, n_passos=1500)

    print(f"== Equilibrio espectral (7 bandas) ==")
    print(f"  T_superficie = {Ts_final:.2f} K   (cinza equivalente: {Ts_gray:.2f} K)\n")

    # OLR por banda no equilibrio final
    F_up_tot, F_dn_tot, F_up_b, F_dn_b = fluxos_totais(T_final, Ts_final, eps_bandas)
    OLR_por_banda = np.array([Fu[-1] for Fu in F_up_b])
    print("== OLR por banda (topo da atmosfera) ==")
    for ib in range(N_BANDAS):
        print(f"  {NOMES_BANDAS[ib].replace(chr(10),' '):20s}: {OLR_por_banda[ib]:6.2f} W/m2")
    print(f"  TOTAL: {OLR_por_banda.sum():.2f} W/m2  (deve ser ~S_abs={S_abs:.2f})\n")

    # Taxas de resfriamento por banda para o perfil tipo-troposfera (fora do equilibrio)
    H = 7.4
    z_centros = -H * np.log(p_centros / p_niveis[0])
    T_inicial = np.maximum(288.0 - 6.5 * z_centros, 200.0)
    _, _, F_up_b0, F_dn_b0 = fluxos_totais(T_inicial, 288.0, eps_bandas)
    dTdt_por_banda = np.array([taxa_aquecimento(Fu, Fd, p_niveis)
                                for Fu, Fd in zip(F_up_b0, F_dn_b0)])

    # ---------------------------- FIGURAS ------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Espessura optica coluna por banda
    ax = axes[0, 0]
    tau_col = dtau_bandas.sum(axis=1)
    ax.bar(range(N_BANDAS), tau_col, color="tab:blue")
    ax.set_xticks(range(N_BANDAS))
    ax.set_xticklabels(NOMES_BANDAS, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"Espessura optica da coluna, $\tau$")
    ax.set_title("Espessura optica por banda\n(janela atmosferica quase transparente)")
    ax.grid(alpha=0.3, axis="y")

    # (b) OLR por banda
    ax = axes[0, 1]
    ax.bar(range(N_BANDAS), OLR_por_banda, color="tab:red")
    ax.set_xticks(range(N_BANDAS))
    ax.set_xticklabels(NOMES_BANDAS, fontsize=8)
    ax.set_ylabel("OLR por banda (W/m2)")
    ax.set_title(f"Emissao para o espaco por banda\n(soma = {OLR_por_banda.sum():.1f} W/m2)")
    ax.grid(alpha=0.3, axis="y")

    # (c) Taxas de resfriamento por banda (perfil tipo-troposfera)
    ax = axes[1, 0]
    for ib in [0, 2, 3, 4, 5]:
        ax.plot(dTdt_por_banda[ib], p_centros, label=NOMES_BANDAS[ib].replace("\n", " "))
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Taxa de resfriamento (K/dia)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("Resfriamento LW por banda\n(janela ~zero, bandas de H2O dominam)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (d) Perfil de equilibrio: cinza (Etapa 2) x espectral (Etapa 5)
    ax = axes[1, 1]
    ax.plot(T_gray, p_centros, "--", label="Cinza (Etapa 2)")
    ax.plot(T_final, p_centros, "-o", ms=3, label="Espectral, 7 bandas (Etapa 5)")
    ax.axvline(Te, color="gray", ls=":", lw=1, label="T_efetiva (Te)")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Temperatura (K)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("Equilibrio radiativo: cinza vs espectral\n(so LW, sem SW na atmosfera)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("./etapa5_bandas_espectrais.png", dpi=150)
    print("Figura salva em etapa5_bandas_espectrais.png")
