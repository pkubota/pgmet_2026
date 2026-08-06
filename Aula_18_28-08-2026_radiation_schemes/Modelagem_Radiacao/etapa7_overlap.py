"""
ETAPA 7 - Sobreposicao de GASES e sobreposicao de NUVENS (McICA)
=====================================================================

Duas fisicas novas nesta etapa, ambas sobre "como combinar coisas que
nao sao homogeneas dentro de uma coluna":

PARTE A - Sobreposicao espectral de gases (gas overlap)
  Na Etapa 6, cada banda tinha UM gas dominante. Na realidade, varias
  bandas tem DOIS (ou mais) gases absorvendo ao mesmo tempo (ex.: a
  banda de 15um do CO2 tambem tem H2O absorvendo). A hipotese padrao
  ("random overlap" espectral, usada no RRTMG/ecRad) e que as linhas dos
  dois gases NAO estao correlacionadas dentro da banda -- entao a
  transmitancia conjunta e obtida combinando os pontos-g de cada gas em
  pares (produto tensorial): cada combinacao (i,j) vira um "ponto-g
  conjunto" com peso w_i*w_j e espessura optica k_A,i*u_A + k_B,j*u_B.
  Validamos isso contra uma integracao Monte Carlo independente.

PARTE B - Sobreposicao de nuvens fracionarias (cloud overlap / McICA)
  Nuvens raramente cobrem 100% de uma coluna. Cada camada tem uma
  FRACAO de nuvem c_i. Como camadas de nuvem se sobrepoem verticalmente
  (aleatorio? maximo? os dois?) muda MUITO a cobertura total de nuvens e
  o albedo medio da coluna. O metodo McICA (usado no RRTMG e no ecRad,
  cf. pratica do pacote ecrad que voce enviou) resolve isso GERANDO
  "sub-colunas" aleatorias (nuvem liga/desliga por camada, respeitando a
  estatistica de sobreposicao) e fazendo a media do calculo radiativo
  (aqui, o two-stream da Etapa 4) sobre essas sub-colunas.

COMO EXECUTAR
--------------
    python3 etapa7_overlap.py

Roda em sequencia a Parte A (overlap espectral de gases, com validacao
Monte Carlo) e a Parte B (overlap de nuvens / McICA). Parametros
editaveis no `if __name__ == "__main__":`: M_co2/M_h2o (caminhos de
massa), perfil de fracao de nuvem `c` (array de 10 camadas), n_sub
(numero de subcolunas McICA).

SAIDA: prints com as duas validacoes (Monte Carlo para overlap espectral;
solucao exata por enumeracao para o albedo McICA), mais a figura
`etapa7_overlap_gases_nuvens.png`.

FUNCOES PRINCIPAIS
--------------------
combina_overlap_2gases(k_A, w_A, k_B, w_B) -> (k_A_exp, k_B_exp, w_comb)
    Produto tensorial dos pontos-g de dois gases (n_A*n_B pontos
    combinados) -- hipotese de nao-correlacao espectral entre os gases.

gera_subcolunas(c, n_sub, esquema, rng) -> mascara booleana (n_sub, N)
    Gerador de subcolunas de nuvem (algoritmo de Raisanen et al. 2004).
    `esquema` in {"aleatorio", "maximo", "maximo-aleatorio"}; `c` e o
    perfil de fracao de nuvem por camada (0 a 1).

camada_two_stream(...) / combina_duas_camadas(...)
    Reaproveitadas da Etapa 4 (definidas de novo aqui para o arquivo ficar
    autocontido).

NOTA: a funcao `albedo_coluna(tem_baixa, tem_media)` usada para a
aplicacao radiativa e definida DENTRO do bloco `if __name__ == "__main__":`
(nao e importavel de fora) -- se for reusar essa logica em outro script,
copie essa funcao para o nivel do modulo primeiro.
"""

import numpy as np
import matplotlib.pyplot as plt

SIGMA = 5.670374419e-8
G = 9.81


# =======================================================================
# PARTE A - SOBREPOSICAO ESPECTRAL DE GASES (correlated-k overlap)
# =======================================================================
def distribuicao_k(k_centro, espalhamento_decadas, n_g=8):
    nos, pesos = np.polynomial.legendre.leggauss(n_g)
    g = 0.5 * (nos + 1.0)
    w = 0.5 * pesos
    k_g = k_centro * 10.0 ** (espalhamento_decadas * (g - 0.5))
    return k_g, w


def combina_overlap_2gases(k_A, w_A, k_B, w_B):
    """Produto tensorial dos pontos-g de dois gases -- hipotese de
    'sobreposicao aleatoria' espectral (linhas dos dois gases nao
    correlacionadas dentro da banda)."""
    n_A, n_B = len(k_A), len(k_B)
    k_A_exp = np.repeat(k_A, n_B)
    k_B_exp = np.tile(k_B, n_A)
    w_comb = np.repeat(w_A, n_B) * np.tile(w_B, n_A)
    return k_A_exp, k_B_exp, w_comb


if __name__ == "__main__":
    print("=" * 70)
    print("PARTE A: sobreposicao espectral CO2 + H2O na banda de 15 um")
    print("=" * 70)

    # Caminhos de massa da coluna inteira (valores tipicos, cf. Etapa 6)
    M_co2, M_h2o = 4.8, 25.75  # kg/m2 (coluna inteira)
    k_centro_co2, espalh_co2 = 5.0, 5.0
    k_centro_h2o, espalh_h2o = 3.0, 4.0   # absorcao secundaria de H2O nessa banda

    k_co2, w_co2 = distribuicao_k(k_centro_co2, espalh_co2, n_g=8)
    k_h2o, w_h2o = distribuicao_k(k_centro_h2o, espalh_h2o, n_g=8)
    k_A_exp, k_B_exp, w_comb = combina_overlap_2gases(k_co2, w_co2, k_h2o, w_h2o)

    T_quad = np.sum(w_comb * np.exp(-(k_A_exp * M_co2 + k_B_exp * M_h2o)))

    # --- Validacao independente: integracao Monte Carlo -----------------
    rng = np.random.default_rng(42)
    n_amostras = 2_000_000
    gA = rng.random(n_amostras)
    gB = rng.random(n_amostras)
    kA_s = k_centro_co2 * 10.0 ** (espalh_co2 * (gA - 0.5))
    kB_s = k_centro_h2o * 10.0 ** (espalh_h2o * (gB - 0.5))
    T_amostras = np.exp(-(kA_s * M_co2 + kB_s * M_h2o))
    T_mc = T_amostras.mean()
    erro_mc = T_amostras.std() / np.sqrt(n_amostras)  # erro padrao da media

    print(f"Transmitancia (quadratura 8x8=64 pontos-g combinados): {T_quad:.6f}")
    print(f"Transmitancia (Monte Carlo, {n_amostras:,} amostras):      {T_mc:.6f} +/- {erro_mc:.6f}")
    print(f"Diferenca: {abs(T_quad - T_mc):.6f} "
          f"({'dentro' if abs(T_quad-T_mc) < 5*erro_mc else 'FORA'} de 5 sigma do MC)\n")

    # Comparacao com a aproximacao ingenua "so CO2" (Etapa 6, ignorando H2O)
    T_so_co2 = np.sum(w_co2 * np.exp(-k_co2 * M_co2))
    print(f"Transmitancia ignorando H2O (so CO2, como na Etapa 6): {T_so_co2:.6f}")
    print(f"  -> incluir a sobreposicao com H2O reduz a transmitancia em "
          f"{100*(T_so_co2-T_quad)/T_so_co2:.1f}%\n")

    # =====================================================================
    # PARTE B: SOBREPOSICAO DE NUVENS (random / maximo / maximo-aleatorio)
    # =====================================================================
    print("=" * 70)
    print("PARTE B: sobreposicao de nuvens fracionarias (McICA)")
    print("=" * 70)

    # Perfil de fracao de nuvem: 2 camadas de nuvem (baixa e media),
    # separadas por uma camada de ceu limpo -- caso classico para testar
    # overlap (camadas ADJACENTES tendem a "maximo", camadas SEPARADAS
    # por ceu limpo tendem a "aleatorio").
    N_CAMADAS = 10
    c = np.zeros(N_CAMADAS)
    c[1] = 0.6   # nuvem baixa
    c[2] = 0.6   # nuvem baixa (mesma camada fisica, 2 niveis -> adjacente)
    c[6] = 0.3   # nuvem media, separada da baixa por ceu limpo

    def gera_subcolunas(c, n_sub, esquema, rng):
        """Gera mascara binaria (n_sub, N_CAMADAS) de nuvem via o metodo
        padrao de Raisanen et al. (2004) para overlap maximo-aleatorio."""
        N = len(c)
        r = np.zeros((n_sub, N))
        r[:, 0] = rng.random(n_sub)
        for i in range(1, N):
            novo = rng.random(n_sub)
            if esquema == "aleatorio":
                r[:, i] = novo
            elif esquema == "maximo":
                r[:, i] = r[:, 0]  # sempre reusa o mesmo sorteio (1 unico r por coluna)
            elif esquema == "maximo-aleatorio":
                # se as duas camadas adjacentes sao nubladas, reusa r (overlap
                # maximo local); caso contrario, sorteia de novo (aleatorio
                # entre grupos de nuvem separados)
                adjacentes_nubladas = (c[i - 1] > 0) & (c[i] > 0)
                r[:, i] = np.where(adjacentes_nubladas, r[:, i - 1], novo)
            else:
                raise ValueError(esquema)
        return r < c[np.newaxis, :]

    rng2 = np.random.default_rng(7)
    n_sub = 20000
    resultados = {}
    for esquema in ["aleatorio", "maximo", "maximo-aleatorio"]:
        mask = gera_subcolunas(c, n_sub, esquema, rng2)
        cobertura_total = mask.any(axis=1).mean()  # fracao de subcolunas com alguma nuvem
        c_medio_por_camada = mask.mean(axis=0)
        resultados[esquema] = dict(mask=mask, cobertura=cobertura_total, c_medio=c_medio_por_camada)
        print(f"  Overlap {esquema:18s}: cobertura total de nuvem = {cobertura_total:.4f}")

    # Limites analiticos conhecidos para os 2 grupos de nuvem (baixa~0.6, media~0.3):
    cobertura_aleatoria_analitica = 1.0 - (1 - 0.6) * (1 - 0.3)  # grupos independentes
    cobertura_maxima_analitica = max(0.6, 0.3)
    print(f"\n  Formula analitica (overlap aleatorio entre os 2 grupos): "
          f"1-(1-0.6)(1-0.3) = {cobertura_aleatoria_analitica:.4f}")
    print(f"  Formula analitica (overlap maximo):                       "
          f"max(0.6,0.3) = {cobertura_maxima_analitica:.4f}\n")

    # Validacao: fracao de nuvem media por camada deve reproduzir c_i
    erro_val = np.max(np.abs(resultados["maximo-aleatorio"]["c_medio"] - c))
    print(f"  Validacao: |fracao media das subcolunas - c prescrito| max = {erro_val:.4f} "
          f"(deve ser pequeno, ~1/sqrt(n_sub)~{1/np.sqrt(n_sub):.4f})\n")

    # =====================================================================
    # APLICACAO RADIATIVA: albedo medio via McICA vs solucao exata (caso
    # aleatorio, onde da para enumerar as 4 combinacoes possiveis)
    # =====================================================================
    def camada_two_stream(tau, omega0, g, D=2.0):
        b = (1.0 - g) / 2.0
        if np.isclose(omega0, 1.0):
            tau_e = (1.0 - g) * tau
            return tau_e / (2.0 + tau_e), 2.0 / (2.0 + tau_e)
        k = np.sqrt((1.0 - omega0) * D * ((1.0 - omega0) * D + 2.0 * omega0 * b))
        gm = (1.0 - omega0) * D / k
        gp, gn = 1.0 + gm, 1.0 - gm
        e2 = np.exp(-2.0 * k * tau)
        denom = gp ** 2 - gn ** 2 * e2
        R = gp * gn * (1.0 - e2) / denom
        T = (gp ** 2 - gn ** 2) * np.exp(-k * tau) / denom
        return R, T

    def combina_duas_camadas(Ra, Ta, Rb, Tb):
        denom = 1.0 - Ra * Rb
        return Ra + (Ta ** 2) * Rb / denom, Ta * Tb / denom

    albedo_sfc = 0.1
    tau_baixa, omega0_baixa, g_baixa = 15.0, 0.9997, 0.85
    tau_media, omega0_media, g_media = 6.0, 0.9997, 0.85
    R_baixa, T_baixa = camada_two_stream(tau_baixa, omega0_baixa, g_baixa)
    R_media, T_media = camada_two_stream(tau_media, omega0_media, g_media)

    def albedo_coluna(tem_baixa, tem_media):
        R, T = albedo_sfc, 0.0
        if tem_media:
            R, T = combina_duas_camadas(R_media, T_media, R, T)
        if tem_baixa:
            R, T = combina_duas_camadas(R_baixa, T_baixa, R, T)
        return R

    # --- solucao exata (overlap aleatorio entre os 2 grupos) ------------
    p_baixa, p_media = 0.6, 0.3
    combinacoes = [(0, 0, (1 - p_baixa) * (1 - p_media)),
                   (1, 0, p_baixa * (1 - p_media)),
                   (0, 1, (1 - p_baixa) * p_media),
                   (1, 1, p_baixa * p_media)]
    albedo_exato_aleatorio = sum(prob * albedo_coluna(bool(b), bool(m)) for b, m, prob in combinacoes)

    # --- McICA (subcolunas geradas acima, so os 2 grupos de nuvem) ------
    mask_baixa = np.zeros(n_sub, dtype=bool)
    mask_media = np.zeros(n_sub, dtype=bool)
    for esquema in ["aleatorio", "maximo", "maximo-aleatorio"]:
        mask = resultados[esquema]["mask"]
        tem_baixa = mask[:, 1]  # camada 1 representa o grupo "nuvem baixa"
        tem_media = mask[:, 6]  # camada 6 representa o grupo "nuvem media"
        albedos_sub = np.array([albedo_coluna(b, m) for b, m in zip(tem_baixa, tem_media)])
        resultados[esquema]["albedo_medio"] = albedos_sub.mean()
        resultados[esquema]["albedo_erro"] = albedos_sub.std() / np.sqrt(n_sub)

    print("== Albedo medio da coluna: McICA vs solucao exata ==")
    print(f"  Solucao exata (overlap aleatorio, 4 combinacoes): {albedo_exato_aleatorio:.4f}")
    for esquema in ["aleatorio", "maximo", "maximo-aleatorio"]:
        r = resultados[esquema]
        print(f"  McICA, overlap {esquema:18s}: {r['albedo_medio']:.4f} +/- {r['albedo_erro']:.4f}")
    print()

    # ---------------------------- FIGURAS ------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Transmitancia CO2 sozinho x CO2+H2O (overlap) vs caminho de CO2
    ax = axes[0, 0]
    Ms = np.linspace(0.1, 15, 40)
    T_co2_only = np.array([np.sum(w_co2 * np.exp(-k_co2 * m)) for m in Ms])
    T_overlap = np.array([np.sum(w_comb * np.exp(-(k_A_exp * m + k_B_exp * M_h2o))) for m in Ms])
    ax.plot(Ms, T_co2_only, "--", label="so CO2 (Etapa 6)")
    ax.plot(Ms, T_overlap, "-", label="CO2 + H2O (overlap, Etapa 7)")
    ax.axvline(M_co2, color="gray", ls=":", label="caminho real de CO2")
    ax.set_xlabel("Caminho de massa de CO2 (kg/m2)")
    ax.set_ylabel("Transmitancia da banda 15 um")
    ax.set_title("Efeito da sobreposicao espectral CO2+H2O")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Mascaras de subcolunas (McICA) para os 3 esquemas de overlap
    for idx, esquema in enumerate(["aleatorio", "maximo", "maximo-aleatorio"]):
        pass  # (visualizado no painel c/d abaixo com barras resumo)

    ax = axes[0, 1]
    n_mostrar = 60
    mask_mostrar = resultados["maximo-aleatorio"]["mask"][:n_mostrar].T
    ax.imshow(mask_mostrar, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Sub-coluna (amostra McICA)")
    ax.set_ylabel("Camada")
    ax.set_title(f"Exemplo de {n_mostrar} sub-colunas McICA\n(overlap maximo-aleatorio)")

    # (c) Cobertura total de nuvem por esquema de overlap
    ax = axes[1, 0]
    esquemas = ["aleatorio", "maximo", "maximo-aleatorio"]
    coberturas = [resultados[e]["cobertura"] for e in esquemas]
    ax.bar(esquemas, coberturas, color=["tab:blue", "tab:orange", "tab:green"])
    ax.axhline(cobertura_aleatoria_analitica, color="tab:blue", ls="--", lw=1)
    ax.axhline(cobertura_maxima_analitica, color="tab:orange", ls="--", lw=1)
    ax.set_ylabel("Cobertura total de nuvem")
    ax.set_title("Cobertura de nuvem: o esquema de overlap importa muito")
    ax.grid(alpha=0.3, axis="y")

    # (d) Albedo medio: McICA (3 esquemas) vs solucao exata (aleatorio)
    ax = axes[1, 1]
    albedos_plot = [resultados[e]["albedo_medio"] for e in esquemas]
    erros_plot = [resultados[e]["albedo_erro"] for e in esquemas]
    ax.bar(esquemas, albedos_plot, yerr=erros_plot, color=["tab:blue", "tab:orange", "tab:green"], capsize=4)
    ax.axhline(albedo_exato_aleatorio, color="k", ls="--", label="exato (aleatorio, 4 combinacoes)")
    ax.set_ylabel("Albedo medio da coluna")
    ax.set_title("Albedo via McICA (Etapa 4 + subcolunas) x solucao exata")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("./etapa7_overlap_gases_nuvens.png", dpi=150)
    print("Figura salva em etapa7_overlap_gases_nuvens.png")
