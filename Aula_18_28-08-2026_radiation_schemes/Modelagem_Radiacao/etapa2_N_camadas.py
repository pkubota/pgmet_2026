"""
ETAPA 2 - Modelo de N camadas (LW), grade de pressao realista e
          diagnostico explicito de taxas de aquecimento/resfriamento
=====================================================================

O que muda em relacao a Etapa 1:
  - Em vez de 1-2 camadas "artificiais", agora distribuimos a espessura
    optica total da atmosfera (tau_total) ao longo de N camadas conforme
    a espessura em PRESSAO de cada uma (Delta_p/ps) -- ou seja, assumimos
    um absorvedor cinza bem misturado por massa (como CO2). Isso da uma
    grade vertical fisicamente sensata (mais optica perto da superficie,
    onde ha mais massa de ar).
  - Calculamos os fluxos LW para um perfil de temperatura QUALQUER (nao
    apenas em equilibrio), o que permite diagnosticar a taxa de
    aquecimento/resfriamento radiativo em K/dia -- Eq. 21.2 do material
    do curso: dT/dt = (g/cp) * dF_net/dp
  - Fazemos a marcha no tempo (Euler explicito) a partir de um perfil
    fora de equilibrio ate a atmosfera relaxar para o equilibrio
    radiativo, e conferimos que o estado final bate com a formula
    fechada da Etapa 1 (modelo_N_camadas_equilibrio).

Ainda nesta etapa: so LW, sem espalhamento, sem absorcao solar na
atmosfera (isso entra na Etapa 3). O perfil de equilibrio puramente LW
gerado aqui vai mostrar um problema classico (temperatura decrescendo
monotonicamente ate o topo, sem uma estratosfera "quente") -- e exatamente
a motivacao para a Etapa 3 introduzir aquecimento solar (ozonio) na
atmosfera.

COMO EXECUTAR
--------------
    python3 etapa2_N_camadas.py

Parametros editaveis no bloco `if __name__ == "__main__":`: N (numero de
camadas), ps/ptop (topo e base da grade de pressao, hPa), tau_total
(espessura optica total da coluna).

SAIDA: prints com a taxa de resfriamento instantanea e a comparacao entre
a marcha no tempo e a formula fechada, mais a figura
`etapa2_N_camadas_taxas_aquecimento.png`.

FUNCOES PRINCIPAIS
--------------------
grade_pressao(N, ps=1013.25, ptop=10.0) -> p_niveis [hPa], tamanho N+1
    Grade de N camadas entre a superficie (indice 0) e o topo (indice N).

espessura_optica_camadas(p_niveis, tau_total) -> (eps, dtau)
    Distribui tau_total entre as camadas proporcionalmente a espessura em
    pressao (absorvedor bem misturado por massa); eps_i = 1 - exp(-dtau_i).

calcula_fluxos(T_camadas, Ts, eps) -> (F_up, F_dn), tamanho N+1 cada
    Fluxos LW para QUALQUER perfil de temperatura (nao precisa estar em
    equilibrio). Recursao de camada sem espalhamento (eq. 15.15 do
    material do curso com omega_0=0).

taxa_aquecimento(F_up, F_dn, p_niveis, cp=1004.0, g=9.81) -> dT/dt [K/dia]
    dT/dt = (g/cp) * (F_net[base] - F_net[topo]) / Delta_p, por camada.

relaxa_para_equilibrio(T0, Ts0, eps, p_niveis, S_abs, dt_s=21600.0,
                        n_passos=1200) -> (T_final, Ts_final, hist_T, hist_Ts)
    Marcha no tempo (Euler explicito) partindo de T0/Ts0 ate o equilibrio
    radiativo; serve para validar a formula fechada de forma independente.
"""

import numpy as np
import matplotlib.pyplot as plt

SIGMA = 5.670374419e-8   # Stefan-Boltzmann, W/m2/K4
G = 9.81                  # m/s2
CP = 1004.0                # J/(kg K), calor especifico do ar seco


def fluxo_solar_absorvido(S0=1361.0, albedo=0.3):
    return S0 * (1.0 - albedo) / 4.0


# ---------------------------------------------------------------------
# 1) GRADE VERTICAL E DISTRIBUICAO DA ESPESSURA OPTICA
# ---------------------------------------------------------------------
def grade_pressao(N, ps=1013.25, ptop=10.0):
    """
    N camadas entre a superficie (ps) e o topo (ptop), em hPa.
    Convencao: indice 0 = nivel da superficie, indice N = topo da atmosfera.
    """
    p_niveis = np.linspace(ps, ptop, N + 1)  # hPa, decrescente com a altura
    return p_niveis


DIFUSIVIDADE = 1.66
"""
Fator de difusividade (adimensional). A transmitancia de FLUXO (integrada
sobre o hemisfero) de uma camada com profundidade optica de ABSORCAO
tau_abs nao e simplesmente exp(-tau_abs) -- isso seria a transmitancia de
um raio na direcao vertical (mu=1). A integracao angular correta (Eq. 4.7
do material do curso) e aproximada por Elsasser (1942) como uma
exponencial com um angulo efetivo (Eq. 4.9):
    T_f(tau_abs) = 2*int_0^1 exp(-tau_abs/mu) mu dmu  ~=  exp(-D*tau_abs)
com D = 1/mu_1 = 1,66 (angulo efetivo de 53 graus). O material do curso
(pag. 106) observa que esse e exatamente o valor usado operacionalmente
pelo RRTMG_LW. Usamos D=1.66 em toda emissividade de camada LW deste
projeto a partir desta etapa.
"""


def espessura_optica_camadas(p_niveis, tau_total, D=DIFUSIVIDADE):
    """
    Distribui a espessura optica total de ABSORCAO (tau_total, no LW)
    entre as N camadas proporcionalmente a espessura em pressao de cada
    uma (absorvedor bem misturado por massa, tipo CO2), e converte para
    emissividade de camada usando o fator de difusividade (Eq. 4.9 do
    material do curso, D=1.66 -- mesmo valor do RRTMG_LW).

    Retorna:
        eps  : emissividade/absortividade de cada camada, 1-exp(-D*dtau)
        dtau : espessura optica de ABSORCAO de cada camada (sem D)
    """
    ps = p_niveis[0]
    dp = p_niveis[:-1] - p_niveis[1:]           # espessura em pressao de cada camada (N,), >0
    dtau = tau_total * dp / ps
    eps = 1.0 - np.exp(-D * dtau)
    return eps, dtau


# ---------------------------------------------------------------------
# 2) FLUXOS LW PARA UM PERFIL DE TEMPERATURA QUALQUER (nao so equilibrio)
# ---------------------------------------------------------------------
def calcula_fluxos(T_camadas, Ts, eps):
    """
    Mesma recursao de camada da Etapa 1 (Eq. 15.15 sem espalhamento),
    mas agora usada para QUALQUER perfil T_camadas (nao precisa estar em
    equilibrio). eps[0] = camada mais proxima da superficie.
    """
    eps = np.asarray(eps, dtype=float)
    N = len(eps)
    t = 1.0 - eps

    F_up = np.zeros(N + 1)
    F_up[0] = SIGMA * Ts ** 4
    for i in range(N):
        F_up[i + 1] = t[i] * F_up[i] + eps[i] * SIGMA * T_camadas[i] ** 4

    F_dn = np.zeros(N + 1)
    F_dn[N] = 0.0
    for i in range(N - 1, -1, -1):
        F_dn[i] = t[i] * F_dn[i + 1] + eps[i] * SIGMA * T_camadas[i] ** 4

    return F_up, F_dn


# ---------------------------------------------------------------------
# 3) TAXA DE AQUECIMENTO/RESFRIAMENTO RADIATIVO (K/dia)
# ---------------------------------------------------------------------
def taxa_aquecimento(F_up, F_dn, p_niveis, cp=CP, g=G):
    """
    dT/dt = (g/cp) * (F_net[base] - F_net[topo]) / Delta_p   [Eq. 21.2]
    F_net = F_up - F_dn (fluxo liquido ascendente).
    p_niveis em hPa (convertido para Pa aqui dentro).
    Retorna a taxa em K/dia.
    """
    F_net = F_up - F_dn                                   # (N+1,)
    dp_pa = (p_niveis[:-1] - p_niveis[1:]) * 100.0          # hPa -> Pa, (N,) > 0
    convergencia = F_net[:-1] - F_net[1:]                   # (N,)
    dTdt_s = (g / (cp * dp_pa)) * convergencia               # K/s
    return dTdt_s * 86400.0                                  # K/dia


# ---------------------------------------------------------------------
# 4) SOLUCAO DE EQUILIBRIO EM FORMA FECHADA (reaproveitada da Etapa 1)
# ---------------------------------------------------------------------
def modelo_N_camadas_equilibrio(S_abs, eps):
    eps = np.asarray(eps, dtype=float)
    N = len(eps)
    t = 1.0 - eps
    F_dn = np.zeros(N + 1)
    for j in range(N, 0, -1):
        F_dn[j - 1] = F_dn[j] + S_abs * eps[j - 1] / (1.0 + t[j - 1])
    x_camadas = np.zeros(N)
    for j in range(N, 0, -1):
        x_camadas[j - 1] = F_dn[j] + S_abs / (1.0 + t[j - 1])
    T_camadas = (x_camadas / SIGMA) ** 0.25
    Ts4 = S_abs + F_dn[0]
    Ts = (Ts4 / SIGMA) ** 0.25
    return T_camadas, Ts, F_dn


# ---------------------------------------------------------------------
# 5) MARCHA NO TEMPO ATE O EQUILIBRIO RADIATIVO
# ---------------------------------------------------------------------
def relaxa_para_equilibrio(T0, Ts0, eps, p_niveis, S_abs,
                            dt_s=6 * 3600.0, n_passos=1200,
                            guarda_historico=True):
    """
    Euler explicito. A superficie e diagnosticada a cada passo (balanco
    de energia instantaneo, sem inercia termica -- equivalente a uma
    capacidade termica superficial muito pequena/skin temperature).
    """
    T = T0.copy()
    Ts = Ts0
    hist_T = [T.copy()] if guarda_historico else None
    hist_Ts = [Ts] if guarda_historico else None

    for _ in range(n_passos):
        F_up, F_dn = calcula_fluxos(T, Ts, eps)
        dTdt = taxa_aquecimento(F_up, F_dn, p_niveis)  # K/dia
        T = T + dTdt * (dt_s / 86400.0)
        Ts = ((S_abs + F_dn[0]) / SIGMA) ** 0.25
        if guarda_historico:
            hist_T.append(T.copy())
            hist_Ts.append(Ts)

    return T, Ts, np.array(hist_T) if guarda_historico else None, \
        np.array(hist_Ts) if guarda_historico else None


def valida_exemplo_modtran():
    """
    Reproduz o EXEMPLO NUMERICO do material do curso (Lecture 12, pag. 6):
    taxa de resfriamento noturno de uma camada de 0 a 1 km, usando fluxos
    IR calculados pelo MODTRAN para a US Standard Atmosphere 1976.

        Altitude (km)   F_up (W/m2)   F_dn (W/m2)
        0               390           285
        1               375           250

    Resultado esperado no material: dT/dt = -1,5 K/dia (-1,7e-5 K/s).
    Aqui recalculamos com a MESMA formula usada em taxa_aquecimento(),
    mas em coordenada z (altura) em vez de p (pressao), ja que o exemplo
    do curso e dado em altura.
    """
    F_up = np.array([390.0, 375.0])   # W/m2, em z=0 e z=1 km
    F_dn = np.array([285.0, 250.0])
    F_net = F_up - F_dn                # 105, 125 W/m2
    dz = 1000.0                        # m
    rho = 1.17                         # kg/m3 (ar perto da superficie)
    cp = 1004.0                        # J/(kg K)
    dTdt_s = -(F_net[1] - F_net[0]) / (rho * cp * dz)   # K/s
    dTdt_dia = dTdt_s * 86400.0
    print("== Validacao contra o exemplo MODTRAN do curso (Lecture 12, pag. 6) ==")
    print(f"  F_net(0km)={F_net[0]:.0f} W/m2   F_net(1km)={F_net[1]:.0f} W/m2")
    print(f"  dT/dt calculado = {dTdt_s:.2e} K/s = {dTdt_dia:.2f} K/dia")
    print(f"  dT/dt esperado (material do curso) = -1.7e-05 K/s = -1.5 K/dia")
    print(f"  Diferenca: {abs(dTdt_dia - (-1.5)):.3f} K/dia\n")


if __name__ == "__main__":
    valida_exemplo_modtran()

    # ---------------- Configuracao do caso ----------------------------
    N = 30
    ps, ptop = 1013.25, 10.0          # hPa
    tau_total = 4.0                    # espessura optica total (LW) -- ajustavel
    S_abs = fluxo_solar_absorvido()
    Te = (S_abs / SIGMA) ** 0.25

    p_niveis = grade_pressao(N, ps, ptop)
    p_centros = 0.5 * (p_niveis[:-1] + p_niveis[1:])
    eps, dtau = espessura_optica_camadas(p_niveis, tau_total)

    print(f"S_abs = {S_abs:.2f} W/m2   Te = {Te:.2f} K")
    print(f"N camadas = {N}, tau_total = {tau_total}")
    print(f"tau por camada: min={dtau.min():.4f}  max={dtau.max():.4f}\n")

    # ---------------- (a) Taxa de aquecimento instantanea -------------
    # Perfil inicial "tipo troposfera": decai com lapse rate de 6.5 K/km
    # a partir de uma Ts plausivel, ate um teto estratosferico isotermico.
    Ts_inicial = 288.0
    H = 7.4  # km, escala de altura aproximada
    z_centros = -H * np.log(p_centros / ps)  # altitude aproximada (km)
    lapse = 6.5  # K/km
    T_inicial = np.maximum(Ts_inicial - lapse * z_centros, 200.0)

    F_up0, F_dn0 = calcula_fluxos(T_inicial, Ts_inicial, eps)
    dTdt0 = taxa_aquecimento(F_up0, F_dn0, p_niveis)
    print("== Taxa de resfriamento/aquecimento LW instantanea (perfil tipo-troposfera) ==")
    print(f"  min = {dTdt0.min():.2f} K/dia   max = {dTdt0.max():.2f} K/dia   media = {dTdt0.mean():.2f} K/dia\n")

    # ---------------- (b) Relaxacao ate o equilibrio radiativo --------
    T_iso0 = np.full(N, Te)  # comeca isotermica, no valor de Te
    Ts_iso0 = Te + 10.0

    T_final, Ts_final, hist_T, hist_Ts = relaxa_para_equilibrio(
        T_iso0, Ts_iso0, eps, p_niveis, S_abs, dt_s=6 * 3600.0, n_passos=1200
    )

    # Comparacao com a solucao fechada (Etapa 1, generalizada para N camadas)
    T_eq_analitico, Ts_eq_analitico, _ = modelo_N_camadas_equilibrio(S_abs, eps)

    erro_T = np.max(np.abs(T_final - T_eq_analitico))
    erro_Ts = abs(Ts_final - Ts_eq_analitico)
    print("== Relaxacao numerica x solucao fechada (equilibrio radiativo) ==")
    print(f"  T_superficie: numerico={Ts_final:.2f} K   analitico={Ts_eq_analitico:.2f} K")
    print(f"  Maior diferenca entre camadas: {erro_T:.4f} K")
    print(f"  Diferenca na superficie:       {erro_Ts:.4f} K\n")

    # ---------------------------- FIGURAS ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Perfil inicial e taxa de aquecimento instantanea
    ax = axes[0]
    ax.plot(dTdt0, p_centros, color="tab:red")
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("dT/dt (K/dia)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("Taxa de resfriamento LW\n(perfil tipo-troposfera, fora do equilibrio)")
    ax.grid(alpha=0.3)

    # (b) Perfil final relaxado x solucao fechada
    ax = axes[1]
    ax.plot(T_final, p_centros, "o-", ms=3, label="Marcha no tempo (numerico)")
    ax.plot(T_eq_analitico, p_centros, "--", label="Solucao fechada (Etapa 1)")
    ax.axvline(Te, color="gray", ls=":", lw=1, label="T_efetiva (Te)")
    ax.invert_yaxis()
    ax.set_xlabel("Temperatura (K)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("Perfil de equilibrio radiativo\n(so LW -- sem SW na atmosfera)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Evolucao temporal (spin-up) de algumas camadas
    ax = axes[2]
    dias = np.arange(hist_T.shape[0]) * (6 * 3600.0) / 86400.0
    for idx, cor in zip([0, N // 2, N - 1], ["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(dias, hist_T[:, idx], color=cor,
                label=f"camada {idx} (p~{p_centros[idx]:.0f} hPa)")
    ax.plot(dias, hist_Ts, color="tab:red", label="superficie")
    ax.set_xlabel("Tempo (dias)")
    ax.set_ylabel("Temperatura (K)")
    ax.set_title("Relaxacao ao equilibrio radiativo\n(marcha no tempo)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("./etapa2_N_camadas_taxas_aquecimento.png", dpi=150)
    print("Figura salva em etapa2_N_camadas_taxas_aquecimento.png")
