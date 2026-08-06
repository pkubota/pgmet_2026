"""
ETAPA 6 - K-DISTRIBUTION / CORRELATED-K
==========================================

Na Etapa 5, tratar cada banda espectral como "cinza" (um unico
coeficiente de absorcao aplicado via lei de Beer) superestimou
violentamente a opacidade de bandas fortes (tau_coluna~700!), porque um
coeficiente medio nao capta que, DENTRO de uma banda larga, ha nucleos de
linha extremamente opacos misturados com asas de linha (e "microjanelas"
entre linhas) quase transparentes. Um feixe atravessando a banda nao
"ve" um meio homogeneo -- ele escapa pelas partes fracas mesmo quando as
partes fortes estao 100% saturadas.

O metodo k-distribution resolve isso approximando a integral espectral
dentro da banda por uma soma ponderada em poucos "pontos-g":

    T_banda(u) = (1/Delta_nu) * integral_banda exp(-k(nu) u) dnu
               = integral_0^1 exp(-k(g) u) dg
               ~= soma_i w_i * exp(-k_i * u)

onde g(k) e a fracao da banda (reordenada por valor de k, do mais fraco
ao mais forte) com coeficiente de absorcao <= k, e k_i, w_i sao pontos e
pesos de quadratura (aqui, Gauss-Legendre com poucos pontos, exatamente
como o RRTMG faz com ~8-16 pontos-g por banda).

A hipotese "correlated-k" e assumir que essa mesma reordenacao g(k) vale
em todas as camadas (mesmo com T, p diferentes) -- ou seja, cada
ponto-g pode ser tratado como um "canal" monocromatico independente que
atravessa toda a coluna, e as camadas se combinam por multiplicacao
simples de transmitancias dentro de cada canal.

COMO EXECUTAR
--------------
    python3 etapa6_correlated_k.py

Parametros de cada banda (K_CENTRO, SIGMA_LN_K, N_G=numero de
pontos-g) estao no topo do arquivo. K_CENTRO usa os coeficientes FORTES
originais (os que saturavam na Etapa 5) -- o objetivo e mostrar que o
k-distribution lida com eles corretamente, sem precisar dilui-los.

SAIDA: prints com a demonstracao 1 (transmitancia ingenua vs
k-distribution), a validacao (espalhamento=0 = resultado cinza-por-banda),
e o equilibrio final; mais a figura `etapa6_correlated_k.png`.

FUNCOES PRINCIPAIS
--------------------
distribuicao_k(k_centro, espalhamento_decadas, n_g=8) -> (k_g, w_g)
    Gera n_g pontos/pesos de quadratura de Gauss-Legendre em g in [0,1],
    mapeados para k(g) = k_centro * 10^(espalhamento*(g-0.5)) -- uma
    distribuicao log-uniforme sintetica de k dentro da banda. Chame com
    espalhamento_decadas=0 para recuperar o caso "cinza" (1 unico k).

calcula_fluxos_banda_kdist(T_camadas, Ts, eps_gi, w_gi, ib) -> (F_up, F_dn)
    Como calcula_fluxos_banda() da Etapa 5, mas vetorizado sobre os n_g
    pontos-g (eps_gi tem forma (n_g, N_camadas)); cada ponto-g contribui
    w_gi * B_banda(T, ib) para a fonte termica.

fluxos_totais_kdist(...) / relaxa_equilibrio_kdist(...)
    Mesma interface das funcoes equivalentes da Etapa 5, agora somando
    tambem sobre os pontos-g de cada banda.
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
# BANDAS (mesma estrutura da Etapa 5), agora com um "espalhamento" (em
# decadas de log10(k)) que descreve o quanto k varia dentro da banda.
# k_centro usa os valores ORIGINAIS (mais fortes, "realistas") da
# primeira tentativa da Etapa 5 -- o k-distribution deve lidar com eles
# sem precisar diluir os coeficientes artificialmente.
# ---------------------------------------------------------------------
BANDAS = [(1, 200), (200, 500), (500, 800), (800, 980),
          (980, 1100), (1100, 1500), (1500, 3000)]
NOMES_BANDAS = ["Far-IR\n(H2O rot.)", "H2O rot.", "CO2 15um",
                "Janela\natmosferica", "O3 9.6um", "H2O vib-rot", "Fraco"]
N_BANDAS = len(BANDAS)

GAS_DOMINANTE = ["h2o", "h2o", "co2", "h2o", "o3", "h2o", "h2o"]
K_CENTRO = np.array([30.0, 10.0, 5.0, 0.05, 100.0, 8.0, 0.5])   # m2/kg
SIGMA_LN_K = np.array([2.2, 2.2, 2.8, 1.1, 1.7, 2.2, 1.1])  # desvio-padrao de ln(k) por banda (log-normal, Eq. 6.1-6.2 do curso)
N_G = 8  # pontos-g por banda (RRTMG usa tipicamente 8-16)
DIFUSIVIDADE = 1.66  # Elsasser (1942), Eq. 4.9 do curso -- mesmo valor do RRTMG_LW


def distribuicao_k(k_centro, sigma_ln, n_g=N_G):
    """
    Pontos e pesos de Gauss-Legendre em g in [0,1], mapeados para k(g)
    assumindo que k e LOG-NORMALMENTE distribuido dentro da banda -- a
    forma de f(k) explicitamente definida no material do curso (Secao
    6.1, "The k-distributions and the correlated-k method", citando Liou
    1992):

        f(k) = 1/(sigma*sqrt(2*pi)) * exp(-(ln(k)-mu)^2 / (2*sigma^2))

    com mu = media(ln k) = ln(k_centro) e sigma = desvio-padrao(ln k).
    A funcao cumulativa g(k) = integral_0^k f(k')dk' (Eq. 6.2 do curso) e
    invertida usando a funcao quantil da normal padrao (scipy.stats.norm)
    para obter k(g), e a integral T(u) = int_0^1 exp(-k(g)u) dg (Eq. 6.3)
    e aproximada pela quadratura de Gauss-Legendre em g.

    NOTA: k_centro e sigma_ln aqui sao ainda escolhidos de forma
    ilustrativa (nao vem de dados espectroscopicos reais como no RRTMG) --
    o que mudou em relacao a versao anterior deste script e a FORMA da
    distribuicao (log-normal, conforme o curso), nao a origem dos valores
    numericos de k_centro.
    """
    from scipy.stats import norm
    nos, pesos = np.polynomial.legendre.leggauss(n_g)
    g = 0.5 * (nos + 1.0)
    w = 0.5 * pesos
    mu = np.log(k_centro)
    k_g = np.exp(mu + sigma_ln * norm.ppf(g))
    return k_g, w


# ---------------------------------------------------------------------
# FUNCAO DE PLANCK POR BANDA (identica a Etapa 5)
# ---------------------------------------------------------------------
def planck_nu(nu_cm, T):
    nu_m = nu_cm * 100.0
    x = H_PLANCK * C_LIGHT * nu_m / (K_BOLTZ * T)
    x = np.clip(x, 1e-10, 700.0)
    return 2.0 * H_PLANCK * C_LIGHT ** 2 * nu_m ** 3 / (np.exp(x) - 1.0)


def fracao_planck_banda(T, nu1, nu2, n=300):
    nu = np.linspace(nu1, nu2, n)
    B = planck_nu(nu, T)
    integral = np.trapezoid(B, nu * 100.0)
    return np.pi * integral / (SIGMA * T ** 4)


_T_TABELA = np.arange(150.0, 341.0, 1.0)
_FRAC_TABELA = np.zeros((N_BANDAS, len(_T_TABELA)))
for _ib, (_nu1, _nu2) in enumerate(BANDAS):
    for _it, _T in enumerate(_T_TABELA):
        _FRAC_TABELA[_ib, _it] = fracao_planck_banda(_T, _nu1, _nu2)
_FRAC_TABELA /= _FRAC_TABELA.sum(axis=0, keepdims=True)


def B_banda(T, ib):
    frac = np.interp(T, _T_TABELA, _FRAC_TABELA[ib])
    return frac * SIGMA * T ** 4


# ---------------------------------------------------------------------
# GRADE E PERFIS DE GASES (identicos a Etapa 5)
# ---------------------------------------------------------------------
def grade_pressao(N, ps=1013.25, ptop=1.0):
    return np.exp(np.linspace(np.log(ps), np.log(ptop), N + 1))


def perfis_gases(p_niveis, H_atm=7.4):
    p_centros = 0.5 * (p_niveis[:-1] + p_niveis[1:])
    ps = p_niveis[0]
    z = -H_atm * np.log(p_centros / ps)
    q_h2o = 0.010 * np.exp(-z / 2.5)
    q_co2 = np.full_like(p_centros, 6.08e-4)
    q_o3 = 8e-6 * np.exp(-0.5 * ((np.log(p_centros) - np.log(10.0)) / 0.7) ** 2)
    return q_h2o, q_co2, q_o3


def caminhos_de_massa(p_niveis, q_h2o, q_co2, q_o3):
    dp_pa = (p_niveis[:-1] - p_niveis[1:]) * 100.0
    return {"h2o": q_h2o * dp_pa / G, "co2": q_co2 * dp_pa / G, "o3": q_o3 * dp_pa / G}


# ---------------------------------------------------------------------
# FLUXOS COM K-DISTRIBUTION (vetorizado sobre os N_g pontos-g)
# ---------------------------------------------------------------------
def calcula_fluxos_banda_kdist(T_camadas, Ts, eps_gi, w_gi, ib):
    """
    eps_gi: array (n_g, N) -- emissividade de cada camada em cada ponto-g
    w_gi:   array (n_g,)   -- peso de quadratura de cada ponto-g
    Retorna F_up, F_dn (N+1,) somados (integrados) sobre os pontos-g.
    """
    n_g, N = eps_gi.shape
    t_gi = 1.0 - eps_gi

    F_up_total = np.zeros(N + 1)
    F_up = w_gi * B_banda(Ts, ib)
    F_up_total[0] = F_up.sum()
    for i in range(N):
        Bi = B_banda(T_camadas[i], ib)
        F_up = t_gi[:, i] * F_up + eps_gi[:, i] * w_gi * Bi
        F_up_total[i + 1] = F_up.sum()

    F_dn_total = np.zeros(N + 1)
    F_dn = np.zeros(n_g)
    for i in range(N - 1, -1, -1):
        Bi = B_banda(T_camadas[i], ib)
        F_dn = t_gi[:, i] * F_dn + eps_gi[:, i] * w_gi * Bi
        F_dn_total[i] = F_dn.sum()

    return F_up_total, F_dn_total


def fluxos_totais_kdist(T_camadas, Ts, eps_gi_bandas, w_gi_bandas):
    N = T_camadas.shape[0]
    F_up_tot, F_dn_tot = np.zeros(N + 1), np.zeros(N + 1)
    for ib in range(N_BANDAS):
        Fu, Fd = calcula_fluxos_banda_kdist(T_camadas, Ts, eps_gi_bandas[ib], w_gi_bandas[ib], ib)
        F_up_tot += Fu
        F_dn_tot += Fd
    return F_up_tot, F_dn_tot


def taxa_aquecimento(F_up, F_dn, p_niveis):
    F_net = F_up - F_dn
    dp_pa = (p_niveis[:-1] - p_niveis[1:]) * 100.0
    conv = F_net[:-1] - F_net[1:]
    return (G / (CP * dp_pa)) * conv * 86400.0


def relaxa_equilibrio_kdist(T0, Ts0, eps_gi_bandas, w_gi_bandas, p_niveis, S_abs,
                             dt_s=6 * 3600.0, n_passos=1500):
    T, Ts = T0.copy(), Ts0
    for _ in range(n_passos):
        F_up, F_dn = fluxos_totais_kdist(T, Ts, eps_gi_bandas, w_gi_bandas)
        dTdt = taxa_aquecimento(F_up, F_dn, p_niveis)
        T = T + dTdt * (dt_s / 86400.0)
        Ts = ((S_abs + F_dn[0]) / SIGMA) ** 0.25
    return T, Ts


if __name__ == "__main__":
    N = 30
    S_abs = fluxo_solar_absorvido()
    Te = (S_abs / SIGMA) ** 0.25
    p_niveis = grade_pressao(N)
    p_centros = 0.5 * (p_niveis[:-1] + p_niveis[1:])
    q_h2o, q_co2, q_o3 = perfis_gases(p_niveis)
    massas = caminhos_de_massa(p_niveis, q_h2o, q_co2, q_o3)

    # ===================================================================
    # DEMONSTRACAO 1: transmitancia da COLUNA INTEIRA, banda far-IR H2O
    # (a banda que mais saturava na Etapa 5) -- Beer's law ingenuo vs
    # k-distribution, para o MESMO k_centro "realista".
    # ===================================================================
    M_total_h2o = massas["h2o"].sum()
    k_c, spread = K_CENTRO[0], SIGMA_LN_K[0]
    T_naive = np.exp(-k_c * M_total_h2o)
    k_g, w_g = distribuicao_k(k_c, spread, N_G)
    T_kdist = np.sum(w_g * np.exp(-k_g * M_total_h2o))

    print("== Demonstracao 1: transmitancia da coluna inteira (banda Far-IR H2O) ==")
    print(f"  Caminho de massa total de H2O: {M_total_h2o:.2f} kg/m2")
    print(f"  tau_centro = k_centro * M = {k_c * M_total_h2o:.1f}  (extremamente saturado)")
    print(f"  Transmitancia (Beer's law ingenuo, Etapa 5): {T_naive:.3e}  (zero em ponto flutuante)")
    print(f"  Transmitancia (k-distribution, {N_G} pontos-g):  {T_kdist:.3e}  (pequena, mas nao-nula)")
    print(f"  -> o k-distribution 'vaza' radiacao pelos pontos-g fracos, "
          f"mesmo com a banda saturada no coeficiente medio.\n")

    # ===================================================================
    # VALIDACAO: espalhamento=0 (todos os pontos-g colapsam em k_centro)
    # deve reproduzir o resultado "cinza" da Etapa 5 (um so coeficiente).
    # ===================================================================
    eps_gi_val, w_gi_val = [], []
    for ib in range(N_BANDAS):
        m = massas[GAS_DOMINANTE[ib]]
        k_g0, w_g0 = distribuicao_k(K_CENTRO[ib], 0.0, N_G)  # espalhamento=0
        dtau0 = k_g0[:, None] * m[None, :]
        eps_gi_val.append(1.0 - np.exp(-DIFUSIVIDADE * dtau0))
        w_gi_val.append(w_g0)

    T0 = np.full(N, Te)
    Ts0 = Te + 10.0
    T_val, Ts_val = relaxa_equilibrio_kdist(T0, Ts0, eps_gi_val, w_gi_val, p_niveis, S_abs, n_passos=800)

    # equivalente "gray-band" direto (Etapa 5, mesmo k_centro, sem g-dist):
    eps_gray_equiv = []
    for ib in range(N_BANDAS):
        m = massas[GAS_DOMINANTE[ib]]
        eps_gray_equiv.append(1.0 - np.exp(-DIFUSIVIDADE * K_CENTRO[ib] * m))

    def fluxos_gray(T_camadas, Ts, eps_bandas):
        Np = len(T_camadas)
        F_up_tot, F_dn_tot = np.zeros(Np + 1), np.zeros(Np + 1)
        for ib in range(N_BANDAS):
            eps = eps_bandas[ib]
            t = 1 - eps
            F_up = np.zeros(Np + 1)
            F_up[0] = B_banda(Ts, ib)
            for i in range(Np):
                F_up[i + 1] = t[i] * F_up[i] + eps[i] * B_banda(T_camadas[i], ib)
            F_dn = np.zeros(Np + 1)
            for i in range(Np - 1, -1, -1):
                F_dn[i] = t[i] * F_dn[i + 1] + eps[i] * B_banda(T_camadas[i], ib)
            F_up_tot += F_up
            F_dn_tot += F_dn
        return F_up_tot, F_dn_tot

    T_gray_ref, Ts_gray_ref = T0.copy(), Ts0
    for _ in range(800):
        Fu, Fd = fluxos_gray(T_gray_ref, Ts_gray_ref, eps_gray_equiv)
        dTdt = taxa_aquecimento(Fu, Fd, p_niveis)
        T_gray_ref = T_gray_ref + dTdt * (6 * 3600.0 / 86400.0)
        Ts_gray_ref = ((S_abs + Fd[0]) / SIGMA) ** 0.25

    print("== Validacao: espalhamento=0 deve reproduzir o tratamento cinza-por-banda ==")
    print(f"  k-distribution (espalhamento=0): T_superficie = {Ts_val:.3f} K")
    print(f"  Cinza-por-banda direto:          T_superficie = {Ts_gray_ref:.3f} K")
    print(f"  Diferenca: {abs(Ts_val - Ts_gray_ref):.4f} K\n")

    # ===================================================================
    # CASO REAL: com o espalhamento correto (k-distribution completo),
    # usando os coeficientes k_centro FORTES (originais) que saturavam
    # violentamente na Etapa 5.
    # ===================================================================
    eps_gi_bandas, w_gi_bandas = [], []
    for ib in range(N_BANDAS):
        m = massas[GAS_DOMINANTE[ib]]
        k_g_b, w_g_b = distribuicao_k(K_CENTRO[ib], SIGMA_LN_K[ib], N_G)
        dtau_b = k_g_b[:, None] * m[None, :]
        eps_gi_bandas.append(1.0 - np.exp(-DIFUSIVIDADE * dtau_b))
        w_gi_bandas.append(w_g_b)

    T_final, Ts_final = relaxa_equilibrio_kdist(
        T0, Ts0, eps_gi_bandas, w_gi_bandas, p_niveis, S_abs, n_passos=1500)

    print("== Equilibrio final: k-distribution completo (coeficientes fortes/realistas) ==")
    print(f"  T_superficie = {Ts_final:.2f} K")
    print(f"  (Mesmos coeficientes fortes, SEM k-distribution [cinza-por-banda]: "
          f"Ts={Ts_gray_ref:.2f} K -- mais exagerado)")

    F_up_final, F_dn_final = fluxos_totais_kdist(T_final, Ts_final, eps_gi_bandas, w_gi_bandas)
    print(f"  OLR total = {F_up_final[-1]:.2f} W/m2 (deve ser ~{S_abs:.2f})\n")

    # ---------------------------- FIGURAS ------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Transmitancia da banda far-IR vs caminho optico: Beer's law x k-dist
    ax = axes[0, 0]
    Ms = np.logspace(-1, 2.5, 60)
    T_naive_curva = np.exp(-k_c * Ms)
    T_kdist_curva = np.array([np.sum(w_g * np.exp(-k_g * m)) for m in Ms])
    ax.plot(Ms, T_naive_curva, "--", label="Beer's law ingenuo (Etapa 5)")
    ax.plot(Ms, T_kdist_curva, "-", label=f"k-distribution ({N_G} pontos-g)")
    ax.axvline(M_total_h2o, color="gray", ls=":", label="caminho real da coluna")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Caminho de massa de H2O (kg/m2)")
    ax.set_ylabel("Transmitancia da banda Far-IR")
    ax.set_title("Por que o k-distribution evita a saturacao artificial")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (b) Distribuicao k(g) para cada banda
    ax = axes[0, 1]
    for ib in range(N_BANDAS):
        k_g_plot, _ = distribuicao_k(K_CENTRO[ib], SIGMA_LN_K[ib], 40)
        g_plot = np.linspace(0, 1, 40)
        ax.plot(g_plot, k_g_plot, label=NOMES_BANDAS[ib].replace("\n", " "))
    ax.set_yscale("log")
    ax.set_xlabel("g (fracao acumulada da banda, ordenada por k)")
    ax.set_ylabel("k(g)  (m2/kg)")
    ax.set_title("Distribuicoes k(g) usadas em cada banda")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) Perfil de equilibrio: k-dist(espalhamento=0) x cinza x k-dist completo
    ax = axes[1, 0]
    ax.plot(T_gray_ref, p_centros, "--", label="Cinza-por-banda (ref.)")
    ax.plot(T_val, p_centros, ":", lw=3, label="k-dist, espalhamento=0 (validacao)")
    ax.plot(T_final, p_centros, "-o", ms=3, label="k-distribution completo")
    ax.axvline(Te, color="gray", ls=":", lw=1, label="T_efetiva (Te)")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Temperatura (K)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("Equilibrio radiativo: efeito do k-distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (d) OLR por banda: cinza-forte (Etapa5 nao-recalibrada) x k-dist
    ax = axes[1, 1]
    OLR_kdist = np.array([Fu[-1] for Fu, _ in
                           [calcula_fluxos_banda_kdist(T_final, Ts_final, eps_gi_bandas[ib], w_gi_bandas[ib], ib)
                            for ib in range(N_BANDAS)]])
    ax.bar(range(N_BANDAS), OLR_kdist, color="tab:green")
    ax.set_xticks(range(N_BANDAS))
    ax.set_xticklabels(NOMES_BANDAS, fontsize=8)
    ax.set_ylabel("OLR por banda (W/m2)")
    ax.set_title(f"OLR por banda, k-distribution completo\n(soma = {OLR_kdist.sum():.1f} W/m2)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("./etapa6_correlated_k.png", dpi=150)
    print("Figura salva em etapa6_correlated_k.png")
