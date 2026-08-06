"""
ETAPA 8 - BENCHMARK CONTRA O RRTMG/ecRad REAL
==================================================

Diferente das etapas anteriores (que resolviam o EQUILIBRIO radiativo de
uma coluna idealizada), aqui fazemos um calculo DIAGNOSTICO: pegamos um
perfil atmosferico REAL (temperatura, umidade, ozonio, CO2) extraido do
ERA5 (coluna a 45N, do arquivo era5slice.nc usado no tutorial pratico do
ecRad) e comparamos o que o NOSSO modelo de bandas + k-distribution
(Etapa 6) calcula para o OLR e o fluxo de LW na superficie, contra o que
o ecRad REAL (compilado a partir do codigo-fonte oficial do ECMWF,
usando gas optics RRTMG/ecCKD) calcula para o MESMO perfil.

Isso isola o teste no que realmente importa: o METODO de transferencia
radiativa (bandas idealizadas + k-distribution sintetico vs. RRTMG com
dados espectroscopicos reais), mantendo o estado atmosferico fixo e
identico nos dois casos.

COMO EXECUTAR
--------------
    python3 etapa8_benchmark.py

PRE-REQUISITO IMPORTANTE: este script LE o arquivo
`/home/pkubota/coluna_45N.npz`, que nao e gerado por ele mesmo -- e
produzido pelo script `etapa8a_extrai_coluna_era5.py` (ver esse arquivo),
que por sua vez precisa da saida real do ecRad (`era5slice_out.nc`) e do
executavel `ecrad` compilado a partir do codigo-fonte oficial do ECMWF
(processo descrito no Capitulo 6 do relatorio -- nao e um "pip install").
Sem esses arquivos, este script nao roda; ele NAO refaz a compilacao do
ecRad nem baixa dados sozinho.

Ordem de execucao completa, a partir do zero:
    1. compilar o ecRad (ver Capitulo 6 do relatorio para o passo a passo,
       incluindo a correcao do bug em radiation_interface.F90);
    2. rodar `./ecrad config.nam era5slice.nc era5slice_out.nc`;
    3. rodar `etapa8a_extrai_coluna_era5.py` para gerar coluna_45N.npz;
    4. rodar este script.

SAIDA: tabela comparando OLR e LW descendente na superficie (nosso
modelo x ecRad real), mais a figura `etapa8_benchmark_rrtmg.png`.

FUNCOES PRINCIPAIS
--------------------
Reaproveita a mesma arquitetura de bandas + k-distribution da Etapa 6
(distribuicao_k, B_banda, calcula_fluxos_banda_kdist), mas aplicada a um
perfil de temperatura REAL e FIXO (nao resolve equilibrio -- calculo
diagnostico) e a caminhos de massa de gases REAIS (extraidos do ERA5, nao
os perfis idealizados das etapas anteriores).
"""

import numpy as np
import matplotlib.pyplot as plt

SIGMA = 5.670374419e-8
G = 9.81
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23

# --- bandas e coeficientes (identicos a Etapa 6) ------------------------
BANDAS = [(1, 200), (200, 500), (500, 800), (800, 980),
          (980, 1100), (1100, 1500), (1500, 3000)]
NOMES_BANDAS = ["Far-IR", "H2O rot.", "CO2 15um", "Janela", "O3 9.6um", "H2O vib-rot", "Fraco"]
N_BANDAS = len(BANDAS)
GAS_DOMINANTE = ["h2o", "h2o", "co2", "h2o", "o3", "h2o", "h2o"]
K_CENTRO = np.array([30.0, 10.0, 5.0, 0.05, 100.0, 8.0, 0.5])
SIGMA_LN_K = np.array([2.2, 2.2, 2.8, 1.1, 1.7, 2.2, 1.1])  # desvio-padrao de ln(k) por banda
N_G = 8
DIFUSIVIDADE = 1.66  # Elsasser (1942), Eq. 4.9 do curso -- mesmo valor do RRTMG_LW


def distribuicao_k(k_centro, sigma_ln, n_g=N_G):
    """K-distribution log-normal (Secao 6.1 do material do curso, citando
    Liou 1992) -- ver docstring completa em etapa6_correlated_k.py."""
    from scipy.stats import norm
    nos, pesos = np.polynomial.legendre.leggauss(n_g)
    g = 0.5 * (nos + 1.0)
    w = 0.5 * pesos
    mu = np.log(k_centro)
    k_g = np.exp(mu + sigma_ln * norm.ppf(g))
    return k_g, w


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


def calcula_fluxos_banda_kdist(T_camadas, Ts, eps_gi, w_gi, ib):
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


if __name__ == "__main__":
    d = np.load('/home/pkubota/coluna_45N.npz')
    p_hl, T_hl = d['p_hl'], d['T_hl']
    q, o3_mmr, co2_vmr = d['q'], d['o3'], d['co2']
    Ts = float(d['Ts_skin'])
    N = len(q)  # 137 camadas

    # descarta o topo espurio (p=0 exato) substituindo por um valor pequeno
    p_hl = np.where(p_hl <= 0, 1e-3, p_hl)

    T_camadas = 0.5 * (T_hl[:-1] + T_hl[1:])   # temperatura media da camada
    dp_pa = (p_hl[1:] - p_hl[:-1]) * 100.0      # espessura em pressao (Pa), > 0

    m_h2o = q * dp_pa / G
    m_co2 = (co2_vmr * 44.01 / 28.97) * dp_pa / G   # vmr -> mmr -> caminho de massa
    m_o3 = o3_mmr * dp_pa / G
    massas = {"h2o": m_h2o, "co2": m_co2, "o3": m_o3}

    print(f"Camadas: {N}   Ts (skin) = {Ts:.2f} K")
    print(f"Caminho total: H2O={m_h2o.sum():.2f} kg/m2  CO2={m_co2.sum():.2f} kg/m2  "
          f"O3={m_o3.sum()*1000:.3f} g/m2\n")

    eps_gi_bandas, w_gi_bandas = [], []
    for ib in range(N_BANDAS):
        m = massas[GAS_DOMINANTE[ib]]
        k_g, w_g = distribuicao_k(K_CENTRO[ib], SIGMA_LN_K[ib], N_G)
        dtau = k_g[:, None] * m[None, :]
        eps_gi_bandas.append(1.0 - np.exp(-DIFUSIVIDADE * dtau))
        w_gi_bandas.append(w_g)

    F_up_tot = np.zeros(N + 1)
    F_dn_tot = np.zeros(N + 1)
    F_up_por_banda = []
    for ib in range(N_BANDAS):
        Fu, Fd = calcula_fluxos_banda_kdist(T_camadas, Ts, eps_gi_bandas[ib], w_gi_bandas[ib], ib)
        F_up_tot += Fu
        F_dn_tot += Fd
        F_up_por_banda.append(Fu)

    OLR_nosso = F_up_tot[-1]
    LWdn_sfc_nosso = F_dn_tot[0]

    print("=" * 70)
    print("COMPARACAO: nosso modelo (bandas + k-distribution idealizado)")
    print("            vs ecRad REAL (RRTMG/ecCKD, codigo-fonte ECMWF)")
    print("=" * 70)
    print(f"{'Quantidade':35s}{'Nosso modelo':>15s}{'ecRad real':>15s}{'Dif (%)':>12s}")
    print(f"{'OLR (W/m2)':35s}{OLR_nosso:15.2f}{float(d['OLR_real']):15.2f}"
          f"{100*(OLR_nosso-float(d['OLR_real']))/float(d['OLR_real']):12.1f}")
    print(f"{'LW descendente na superficie (W/m2)':35s}{LWdn_sfc_nosso:15.2f}"
          f"{float(d['LWdn_sfc_real']):15.2f}"
          f"{100*(LWdn_sfc_nosso-float(d['LWdn_sfc_real']))/float(d['LWdn_sfc_real']):12.1f}")
    print()
    print("(SW nao comparado aqui -- nosso modelo de bandas idealizado da Etapa 5/6 "
          "trata so o LW; a Etapa 4 tem o motor de espalhamento SW mas nao foi\n"
          " integrado ao tratamento espectral completo.)\n")

    # ---------------------------- FIGURAS ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(F_up_tot, p_hl, label="Nosso modelo (F_up)")
    ax.plot(d['flux_up_lw_clear'], p_hl, "--", label="ecRad real (F_up)")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Fluxo LW ascendente (W/m2)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("Perfil de fluxo LW ascendente\ncoluna real 45N (ceu limpo)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    categorias = ["OLR\n(topo)", "LW descendente\n(superficie)"]
    nosso = [OLR_nosso, LWdn_sfc_nosso]
    real = [float(d['OLR_real']), float(d['LWdn_sfc_real'])]
    x = np.arange(2)
    ax.bar(x - 0.18, nosso, width=0.35, label="Nosso modelo (Etapa 6)")
    ax.bar(x + 0.18, real, width=0.35, label="ecRad real (RRTMG/ecCKD)")
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    ax.set_ylabel("W/m2")
    ax.set_title("Nosso modelo vs ecRad real\n(mesmo perfil atmosferico ERA5, 45N)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("./etapa8_benchmark_rrtmg.png", dpi=150)
    print("Figura salva em etapa8_benchmark_rrtmg.png")
