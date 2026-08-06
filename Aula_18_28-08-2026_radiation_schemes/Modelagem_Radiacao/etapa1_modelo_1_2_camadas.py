"""
ETAPA 1 - Modelo de Transferencia Radiativa de Onda Longa (LW)
Atmosfera cinza, sem espalhamento -- modelo de 1 e 2 (e N) camadas
===================================================================

Curso: Modelagem de Radiacao (baseado em Liou / AT622 - Stephens)

Hipoteses desta etapa:
- So onda longa (LW); onda curta (SW) entra apenas como um aquecimento fixo
  na superficie (S_abs), sem absorcao pela atmosfera (isso sera introduzido
  na Etapa 3).
- Camadas "cinzas": emissividade/absortividade eps_i nao depende do
  comprimento de onda (sera substituido por bandas espectrais na Etapa 5+).
- Sem espalhamento (essa fisica entra na Etapa 4, com o modelo two-stream
  completo -- Secao 15 do material do curso). Aqui cada camada so absorve e
  emite, como na equacao de Schwarzschild integrada (Lecture 12 do material).
- Equilibrio radiativo puro: em regime estacionario, sem conveccao nem
  absorcao solar na atmosfera, o fluxo liquido de LW e CONSTANTE com a
  altura e igual a radiacao solar absorvida (S_abs = OLR). Essa e a
  condicao que resolvemos analiticamente abaixo.

Equacao de camada (analoga a Eq. 15.15 do material do curso, mas sem termo
de retroespalhamento r, pois aqui omega_0 = 0, isto e, so absorcao+emissao):
    F_up[topo]  = t * F_up[base]  + eps * sigma*T^4
    F_dn[base]  = t * F_dn[topo]  + eps * sigma*T^4
onde t = 1 - eps e a transmissividade da camada em LW.

FUNDAMENTACAO NO MATERIAL DO CURSO (slide "Balanco de Radiacao",
Transferencia Radiativa do MCGA, paginas 10-13): o modelo de 1 camada
implementado abaixo e EXATAMENTE o balanco de energia dali. O material
resolve o mesmo sistema de 3 equacoes usado em modelo_1_camada_analitico:
    Topo:        (1-eps_A)*sigma*Ts^4 + eps_A*sigma*TA^4 = S/4*(1-alpha)
    Atmosfera:   eps_A*sigma*Ts^4 = 2*eps_A*sigma*TA^4
    Superficie:  S/4*(1-alpha) + eps_A*sigma*TA^4 = sigma*Ts^4
O slide usa o caso numerico alpha=0.3, eps_A=0.8 como exemplo ilustrativo
(reproduzido na funcao valida_exemplo_slide() abaixo). Essas equacoes, por
sua vez, sao a forma integrada da equacao de transferencia radiativa geral
(Eq. 2.11 do material) para o caso de 1 camada isotermica, sem
espalhamento e em equilibrio -- a mesma equacao que reaparece ao longo de
todo o curso (Consideracoes, pag. 16-18) antes de ser reduzida a
aproximacao de dois fluxos.

COMO EXECUTAR
--------------
    python3 etapa1_modelo_1_2_camadas.py

Sem argumentos de linha de comando. Os parametros fisicos (S0, albedo,
emissividades testadas) estao no bloco `if __name__ == "__main__":`, no
final do arquivo -- edite os valores diretamente ali para testar outros
cenarios.

SAIDA: prints no terminal com Ts/Ta para cada caso testado, e a figura
`etapa1_modelo_1_2_camadas.png` salva em /mnt/user-data/outputs/.

FUNCOES PRINCIPAIS (podem ser importadas por outros scripts)
--------------------------------------------------------------
fluxo_solar_absorvido(S0=1361.0, albedo=0.3) -> S_abs [W/m2]
    Fluxo solar medio absorvido pelo planeta: S0*(1-albedo)/4.

modelo_1_camada_analitico(S_abs, eps_a) -> (Ts, Ta) [K]
    Solucao fechada para 1 camada cinza de emissividade eps_a (Eq. no
    cabecalho da funcao). Caso especial eps_a=1 recupera o resultado
    classico Ts = 2^(1/4) * Te.

modelo_N_camadas_equilibrio(S_abs, eps) -> (T_camadas, Ts, F_dn)
    Solucao fechada generalizada para N camadas com emissividades
    arbitrarias (array `eps`, indice 0 = camada mais proxima da
    superficie). Nao usa iteracao numerica.
"""

import numpy as np
import matplotlib.pyplot as plt

SIGMA = 5.670374419e-8  # constante de Stefan-Boltzmann, W/m2/K4


def fluxo_solar_absorvido(S0=1361.0, albedo=0.3):
    """Fluxo solar medio absorvido pelo planeta (W/m2): media sobre a
    esfera (fator 1/4) menos a fracao refletida pelo albedo."""
    return S0 * (1.0 - albedo) / 4.0


# ---------------------------------------------------------------------
# 1) SOLUCAO ANALITICA -- 1 camada, emissividade geral eps_a
# ---------------------------------------------------------------------
def modelo_1_camada_analitico(S_abs, eps_a):
    """
    Balanco de um planeta com 1 camada atmosferica cinza (emissividade
    eps_a), transparente a SW.

    Derivacao (balanco de energia):
      Topo:       (1-eps_a)*sig*Ts^4 + eps_a*sig*Ta^4 = S_abs
      Camada:     eps_a*sig*Ts^4 = 2*eps_a*sig*Ta^4   => Ts^4 = 2*Ta^4
      Superficie: sig*Ts^4 = S_abs + eps_a*sig*Ta^4
      => sig*Ts^4*(1 - eps_a/2) = S_abs
    """
    Ts4 = S_abs / (SIGMA * (1.0 - eps_a / 2.0))
    Ta4 = Ts4 / 2.0
    return Ts4 ** 0.25, Ta4 ** 0.25


# ---------------------------------------------------------------------
# 2) SOLUCAO ANALITICA GERAL -- N camadas, emissividade eps[i] arbitraria
#    (funciona para N=1, N=2, ... generaliza direto para a Etapa 2)
# ---------------------------------------------------------------------
def modelo_N_camadas_equilibrio(S_abs, eps):
    """
    eps: lista/array com N emissividades; eps[0] = camada mais proxima da
         superficie ... eps[N-1] = camada mais alta (topo da atmosfera).

    Usa a condicao de equilibrio radiativo puro (fluxo liquido de LW
    constante com a altura, igual a S_abs) para obter uma formula fechada,
    sem iteracao numerica.

    Retorna:
        T_camadas : array (N,) com a temperatura de cada camada
        Ts        : temperatura da superficie
        F_dn      : fluxo descendente de LW em cada um dos N+1 niveis
                     (F_dn[N] = 0 no topo da atmosfera; F_dn[0] chega na
                     superficie)
    """
    eps = np.asarray(eps, dtype=float)
    N = len(eps)
    t = 1.0 - eps  # transmissividade de cada camada em LW

    # Fluxo descendente, calculado do topo (nivel N) para a base (nivel 0):
    F_dn = np.zeros(N + 1)
    F_dn[N] = 0.0
    for j in range(N, 0, -1):
        F_dn[j - 1] = F_dn[j] + S_abs * eps[j - 1] / (1.0 + t[j - 1])

    # Temperatura de cada camada (usando x = sigma*T^4):
    x_camadas = np.zeros(N)
    for j in range(N, 0, -1):
        x_camadas[j - 1] = F_dn[j] + S_abs / (1.0 + t[j - 1])
    T_camadas = (x_camadas / SIGMA) ** 0.25

    # Superficie: balanco de energia (absorve SW direta + LW descendente
    # vindo da primeira camada)
    Ts4 = S_abs + F_dn[0]
    Ts = (Ts4 / SIGMA) ** 0.25

    return T_camadas, Ts, F_dn


def valida_exemplo_slide():
    """
    Reproduz o EXEMPLO NUMERICO do slide "Balanco de Radiacao" do
    material do curso (Transferencia Radiativa do MCGA, pag. 13):
    com alpha=0.3 e eps_A=0.8, o slide encontra Ts=289 K e TA=243 K.
    """
    S_abs = fluxo_solar_absorvido(S0=1361.0, albedo=0.3)
    Ts, Ta = modelo_1_camada_analitico(S_abs, eps_a=0.8)
    print("== Validacao contra o exemplo numerico do slide (pag. 13 do curso) ==")
    print(f"  Calculado aqui:      Ts = {Ts:.2f} K   TA = {Ta:.2f} K")
    print(f"  Valor do slide:      Ts = 289 K        TA = 243 K")
    print(f"  Diferenca:           dTs = {abs(Ts-289):.2f} K   dTA = {abs(Ta-243):.2f} K\n")


if __name__ == "__main__":
    valida_exemplo_slide()
    S_abs = fluxo_solar_absorvido(S0=1361.0, albedo=0.3)
    Te = (S_abs / SIGMA) ** 0.25
    print(f"Fluxo solar absorvido (S_abs): {S_abs:.2f} W/m2")
    print(f"Temperatura efetiva de emissao do planeta (Te): {Te:.2f} K\n")

    # --- Caso classico: 1 camada totalmente opaca (eps_a = 1) -----------
    Ts_1, Ta_1 = modelo_1_camada_analitico(S_abs, eps_a=1.0)
    print("== Modelo de 1 camada (opaca, eps=1) -- solucao analitica ==")
    print(f"  T_superficie = {Ts_1:.2f} K")
    print(f"  T_camada     = {Ta_1:.2f} K\n")

    # Verificacao cruzada usando a formula geral de N camadas:
    T_camadas, Ts_num, _ = modelo_N_camadas_equilibrio(S_abs, eps=[1.0])
    print("  Verificacao cruzada (formula geral p/ N camadas):")
    print(f"  T_superficie = {Ts_num:.2f} K   T_camada = {T_camadas[0]:.2f} K\n")

    # --- 1 camada, emissividade parcial (eps=0.6) ------------------------
    Ts_06, Ta_06 = modelo_1_camada_analitico(S_abs, eps_a=0.6)
    T_camadas_06, Ts_num_06, _ = modelo_N_camadas_equilibrio(S_abs, eps=[0.6])
    print("== Modelo de 1 camada, eps=0.6 (efeito estufa parcial) ==")
    print(f"  Analitico:  T_superficie = {Ts_06:.2f} K   T_camada = {Ta_06:.2f} K")
    print(f"  Formula N:  T_superficie = {Ts_num_06:.2f} K   T_camada = {T_camadas_06[0]:.2f} K\n")

    # --- 2 camadas opacas (eps=1 cada) -----------------------------------
    T_camadas2, Ts_2, F_dn2 = modelo_N_camadas_equilibrio(S_abs, eps=[1.0, 1.0])
    print("== Modelo de 2 camadas opacas (eps=1 cada) ==")
    print(f"  T_superficie      = {Ts_2:.2f} K")
    print(f"  T_camada inferior = {T_camadas2[0]:.2f} K")
    print(f"  T_camada superior = {T_camadas2[1]:.2f} K")
    print(f"  (esperado pela teoria classica: Ts=3^0.25*Te={3**0.25*Te:.2f} K, "
          f"camada inf.=2^0.25*Te={2**0.25*Te:.2f} K, camada sup.=Te={Te:.2f} K)\n")

    # --- 2 camadas com emissividades diferentes (mais realista) ----------
    T_camadas2b, Ts_2b, _ = modelo_N_camadas_equilibrio(S_abs, eps=[0.9, 0.4])
    print("== Modelo de 2 camadas, eps=[0.9 (baixa), 0.4 (alta)] ==")
    print(f"  T_superficie      = {Ts_2b:.2f} K")
    print(f"  T_camada inferior = {T_camadas2b[0]:.2f} K")
    print(f"  T_camada superior = {T_camadas2b[1]:.2f} K\n")

    # --- Sensibilidade: 1 camada com eps_a variavel -----------------------
    eps_vals = np.linspace(0.05, 1.0, 40)
    Ts_vals, Ta_vals = [], []
    for e in eps_vals:
        ts, ta = modelo_1_camada_analitico(S_abs, e)
        Ts_vals.append(ts)
        Ta_vals.append(ta)

    # ---------------------------- FIGURAS --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Sensibilidade Ts/Ta a eps_a
    ax = axes[0]
    ax.plot(eps_vals, Ts_vals, label="T_superficie", lw=2)
    ax.plot(eps_vals, Ta_vals, label="T_camada", lw=2)
    ax.axhline(Te, color="gray", ls="--", lw=1, label="T_efetiva (Te)")
    ax.set_xlabel(r"Emissividade da camada, $\varepsilon_a$")
    ax.set_ylabel("Temperatura (K)")
    ax.set_title("Modelo de 1 camada: sensibilidade ao efeito estufa")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Perfil vertical: 1 camada x 2 camadas (opacas)
    ax = axes[1]
    niveis_1 = [0, 1]
    temps_1 = [Ts_1, Ta_1]
    niveis_2 = [0, 1, 2]
    temps_2 = [Ts_2, T_camadas2[0], T_camadas2[1]]
    ax.step(temps_1, niveis_1, where="post", marker="o", label="1 camada (eps=1)")
    ax.step(temps_2, niveis_2, where="post", marker="s", label="2 camadas (eps=1)")
    ax.axvline(Te, color="gray", ls="--", lw=1, label="T_efetiva (Te)")
    ax.set_xlabel("Temperatura (K)")
    ax.set_ylabel("Nivel (0 = superficie)")
    ax.set_title("Perfil vertical: 1 vs 2 camadas opacas")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("etapa1_modelo_1_2_camadas.png", dpi=150)
    print("Figura salva em etapa1_modelo_1_2_camadas.png")
