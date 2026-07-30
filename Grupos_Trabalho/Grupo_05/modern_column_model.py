"""
================================================================================
MODELO DE COLUNA ACOPLADO OCEANO-ATMOSFERA COM FLUXOS TURBULENTOS REAIS
Perfis verticais (multi-nivel) evoluindo por 72 horas - dados sinteticos
================================================================================

Modernizacao em relacao ao modelo de 2 caixas anterior:

1) FLUXOS DE SUPERFICIE: no lugar do fluxo linearizado F=lambda*(Ta-To), os
   fluxos turbulentos de superficie sao calculados por formulas aerodinamicas
   "bulk":
        H  = rho_ar * cp_ar * Ch * U * (Ts - Ta)      [calor sensivel, W/m2]
        LE = rho_ar * L_v   * Ce * U * (qs(Ts) - qa)   [calor latente, W/m2]
        tau= rho_ar * Cd * U^2                          [tensao de vento, N/m2]

   Dois fechamentos para os coeficientes de transferencia (Ch, Ce, Cd) sao
   implementados e comparados lado a lado, no mesmo espirito da comparacao
   Esquema A / Esquema B do estudo anterior:

     (a) "bulk simples": Ch=Ce=Cd=constante (sem correcao de estabilidade).
     (b) "Monin-Obukhov" (Businger-Dyer/Paulson 1970, ao estilo Louis 1979):
         coeficientes neutros corrigidos iterativamente pela funcao de
         estabilidade phi(z/L), onde L e o comprimento de Obukhov -
         reduzindo a mistura em condicoes estaveis (noite) e aumentando-a
         em condicoes instaveis/convectivas (dia).

2) PERFIS VERTICAIS: cada coluna deixa de ser uma unica caixa e passa a ter
   N niveis verticais, evoluindo por difusao turbulenta:
        dT/dt = (1/rho/cp) * d/dz( rho*cp*Kz * dT/dz ) + fontes
   com Kz diagnosticado a cada passo de tempo a partir da estabilidade
   (atmosfera, perfil tipo O'Brien 1970) ou da tensao de vento (oceano,
   mistura reforcada perto da superficie, no espirito de Kraus & Turner
   1967 / Price, Weller & Pinkel 1986), permitindo visualizar o
   desenvolvimento da camada limite atmosferica e da camada de mistura
   oceanica ao longo do ciclo diurno.

Forcante sintetica (72 h, 3 ciclos diurnos):
   - radiacao solar de onda curta: pico ~700 W/m2 ao meio-dia, zero a noite
   - perda liquida de onda longa: constante, ~60 W/m2 (simplificacao)
   - vento: varia suavemente de ~3 m/s (noite) a ~7 m/s (tarde)

Autor: material de apoio - MET-579 (Grupo 5: acoplamento oceano-atmosfera)
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. CONSTANTES FISICAS
# ------------------------------------------------------------------
g = 9.81
k_von = 0.4                 # constante de von Karman
rho_air = 1.2                # kg/m3
cp_air = 1004.0               # J/kg/K
L_v = 2.5e6                  # J/kg (calor latente de vaporizacao)
rho_water = 1025.0             # kg/m3
cp_water = 4000.0              # J/kg/K (agua do mar)
p_surf = 1013.0               # hPa

# ------------------------------------------------------------------
# 1. GRADES VERTICAIS (uniformes, para simplicidade)
# ------------------------------------------------------------------
Nz_a, Ztop_a = 20, 2000.0        # atmosfera: 20 niveis ate 2000 m
Nz_o, Ztop_o = 20, 100.0          # oceano: 20 niveis ate 100 m

dz_a = Ztop_a / Nz_a
dz_o = Ztop_o / Nz_o
z_a = (np.arange(Nz_a) + 0.5) * dz_a     # centros de celula (atmosfera, m acima da sup.)
z_o = (np.arange(Nz_o) + 0.5) * dz_o     # centros de celula (oceano, m abaixo da sup.)

z_ref = z_a[0]   # altura de referencia para as formulas bulk (~50 m; ver nota no relatorio)

# ------------------------------------------------------------------
# 2. UMIDADE DE SATURACAO (formula de Tetens)
# ------------------------------------------------------------------
def qsat(T_K, p_hPa=p_surf):
    Tc = T_K - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))   # hPa
    return 0.622 * es / (p_hPa - 0.378 * es)          # kg/kg


# ------------------------------------------------------------------
# 3. FLUXOS DE SUPERFICIE - DOIS FECHAMENTOS
# ------------------------------------------------------------------
def fluxes_bulk_simples(Ta, qa, Ts, U, Cd=1.3e-3, Ch=1.3e-3, Ce=1.3e-3):
    """Fechamento (a): coeficientes de transferencia CONSTANTES, sem
    correcao de estabilidade."""
    H = rho_air * cp_air * Ch * U * (Ts - Ta)
    LE = rho_air * L_v * Ce * U * (qsat(Ts) - qa)
    tau = rho_air * Cd * U ** 2
    return H, LE, tau, Cd, Ch, Ce, np.nan  # np.nan no lugar de L_MO (nao aplicavel)


def fluxes_monin_obukhov(Ta, qa, Ts, U, z=z_ref, z0=1.5e-4, z0h=1.5e-5, n_iter=6):
    """Fechamento (b): teoria de similaridade de Monin-Obukhov, com as
    funcoes de estabilidade de Businger-Dyer/Paulson (1970), resolvidas
    iterativamente (semelhante em espirito ao esquema de Louis 1979,
    amplamente usado em modelos operacionais)."""
    Cd_n = (k_von / np.log(z / z0)) ** 2
    Ch_n = k_von ** 2 / (np.log(z / z0) * np.log(z / z0h))
    Ce_n = Ch_n

    u_star = np.sqrt(Cd_n) * max(U, 0.1)
    theta_star = Ch_n * (Ta - Ts) * U / u_star
    theta_v = 0.5 * (Ta + Ts)  # aproximacao simples (ignora efeito da umidade)

    for _ in range(n_iter):
        L_MO = u_star ** 2 * theta_v / (k_von * g * (theta_star + 1e-12))
        zeta = np.clip(z / L_MO, -5, 5)

        if zeta < 0:  # instavel - Paulson (1970)
            x = (1 - 16 * zeta) ** 0.25
            psi_m = 2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2) - 2 * np.arctan(x) + np.pi / 2
            psi_h = 2 * np.log((1 + x ** 2) / 2)
        else:  # estavel - forma linear (Businger et al. 1971)
            psi_m = -5 * zeta
            psi_h = -5 * zeta

        Cd = (k_von / (np.log(z / z0) - psi_m)) ** 2
        Ch = k_von ** 2 / ((np.log(z / z0) - psi_m) * (np.log(z / z0h) - psi_h))
        Ce = Ch

        u_star = np.sqrt(Cd) * max(U, 0.1)
        theta_star = Ch * (Ta - Ts) * U / u_star

    H = rho_air * cp_air * Ch * U * (Ts - Ta)
    LE = rho_air * L_v * Ce * U * (qsat(Ts) - qa)
    tau = rho_air * Cd * U ** 2
    return H, LE, tau, Cd, Ch, Ce, L_MO


# ------------------------------------------------------------------
# 4. DIFUSIVIDADES TURBULENTAS DIAGNOSTICAS
# ------------------------------------------------------------------
def Kz_atmosfera(H_flux, tau, z=z_a):
    """Perfil de Kz tipo O'Brien (1970): cresce da superficie, decai para
    zero no topo da CLP; CLP mais rasa (estavel) ou mais profunda
    (convectiva) conforme o sinal/magnitude do fluxo de calor sensivel."""
    u_star = np.sqrt(max(tau, 1e-6) / rho_air)
    if H_flux > 0:  # instavel / convectivo (superficie mais quente que o ar)
        h_pbl = 200.0 + 1300.0 * np.clip(H_flux / 250.0, 0, 1)
    else:  # estavel (superficie mais fria - tipico a noite)
        h_pbl = 150.0
    Kz = k_von * u_star * z * np.clip(1 - z / h_pbl, 0, None) ** 2
    Kz = np.clip(Kz, 0.05, 50.0)   # fundo/teto numerico
    Kz[z > h_pbl] = 0.05
    return Kz, h_pbl


def Kz_oceano(tau, z=z_o, H_mix_decay=25.0, Kz_wind=8.0e-3, Kz_bg=1.0e-5):
    """Mistura reforcada perto da superficie pelo atrito do vento
    (u*_agua), decaindo com a profundidade - representacao simplificada,
    no espirito de Kraus & Turner (1967)/Price-Weller-Pinkel (1986)."""
    u_star_o = np.sqrt(max(tau, 1e-6) / rho_water)
    Kz = Kz_bg + Kz_wind * u_star_o * np.exp(-z / H_mix_decay)
    return np.clip(Kz, Kz_bg, 5e-2)


# ------------------------------------------------------------------
# 5. SOLVER DE DIFUSAO VERTICAL IMPLICITO (Euler implicito, grade uniforme)
# ------------------------------------------------------------------
def diffuse_implicit(T, Kz_face, dt, dz):
    """Um passo de Euler implicito para dT/dt = d/dz(Kz dT/dz), grade
    uniforme, condicao de contorno de fluxo nulo no topo e na base do
    dominio (fluxos de superficie/fundo sao adicionados manualmente FORA
    desta funcao, como termos-fonte explicitos - Kz_face tem N-1 valores,
    nas interfaces entre niveis)."""
    N = len(T)
    r = dt / dz ** 2
    a = np.zeros(N)  # subdiagonal
    b = np.zeros(N)  # diagonal
    c = np.zeros(N)  # superdiagonal
    d = T.copy()

    for i in range(N):
        Kd = Kz_face[i - 1] if i > 0 else 0.0
        Ku = Kz_face[i] if i < N - 1 else 0.0
        a[i] = -r * Kd
        c[i] = -r * Ku
        b[i] = 1 + r * (Kd + Ku)

    # Thomas algorithm (tridiagonal)
    cp_ = np.zeros(N)
    dp_ = np.zeros(N)
    cp_[0] = c[0] / b[0]
    dp_[0] = d[0] / b[0]
    for i in range(1, N):
        m = b[i] - a[i] * cp_[i - 1]
        cp_[i] = c[i] / m
        dp_[i] = (d[i] - a[i] * dp_[i - 1]) / m
    Tnew = np.zeros(N)
    Tnew[-1] = dp_[-1]
    for i in range(N - 2, -1, -1):
        Tnew[i] = dp_[i] - cp_[i] * Tnew[i + 1]
    return Tnew


# ------------------------------------------------------------------
# 6. FORCANTE SINTETICA (72 h, 3 ciclos diurnos)
# ------------------------------------------------------------------
def forcing(t_hours):
    hour_of_day = t_hours % 24.0
    solar = 700.0 * max(0.0, np.sin(np.pi * (hour_of_day - 6.0) / 12.0)) if 6 <= hour_of_day <= 18 else 0.0
    LW_net = 60.0  # perda liquida de onda longa (W/m2), simplificado constante
    U = 3.0 + 4.0 * np.clip(np.sin(np.pi * (hour_of_day - 8.0) / 14.0), 0, None)
    return solar, LW_net, U


# atenuacao da radiacao solar na coluna de agua (tipo Jerlov, aguas claras)
eta_solar = 15.0  # m


def solar_heating_profile(Q_surf, z_edges):
    """Retorna o aquecimento (W/m2) ABSORVIDO em cada camada oceanica,
    dada a radiacao que penetra ate cada interface (lei de Beer-Lambert)."""
    Q_at_edges = Q_surf * np.exp(-z_edges / eta_solar)
    return Q_at_edges[:-1] - Q_at_edges[1:]  # absorvido por camada (W/m2)


z_o_edges = np.arange(Nz_o + 1) * dz_o  # interfaces (0, dz_o, 2dz_o, ..., Ztop_o)

# ------------------------------------------------------------------
# 7. INTEGRACAO TEMPORAL (72 h) - PARA CADA FECHAMENTO TURBULENTO
# ------------------------------------------------------------------
dt = 300.0          # s (5 min)
total_hours = 72.0
n_steps = int(total_hours * 3600 / dt)


def run_simulation(scheme="mo"):
    # perfis iniciais
    Ta = 297.0 - 0.0098 * z_a          # atmosfera: gradiente quase-adiabatico
    qa = 0.012 * np.exp(-z_a / 2000.0)  # umidade especifica (kg/kg)
    To = 300.0 - 8.0 * np.clip((z_o - 30.0) / 70.0, 0, None)  # camada de mistura + termoclina

    hist = {"t": [], "Ta": [], "qa": [], "To": [], "H": [], "LE": [], "tau": [],
            "Cd": [], "Ch": [], "L_MO": [], "U": [], "solar": [], "h_pbl": []}

    for n in range(n_steps + 1):
        t_hours = n * dt / 3600.0
        solar, LW_net, U = forcing(t_hours)

        Ts = To[0]  # temperatura de "pele" aproximada = nivel superior do oceano
        if scheme == "simples":
            H, LE, tau, Cd, Ch, Ce, L_MO = fluxes_bulk_simples(Ta[0], qa[0], Ts, U)
        else:
            H, LE, tau, Cd, Ch, Ce, L_MO = fluxes_monin_obukhov(Ta[0], qa[0], Ts, U)

        hist["t"].append(t_hours)
        hist["Ta"].append(Ta.copy())
        hist["qa"].append(qa.copy())
        hist["To"].append(To.copy())
        hist["H"].append(H)
        hist["LE"].append(LE)
        hist["tau"].append(tau)
        hist["Cd"].append(Cd)
        hist["Ch"].append(Ch)
        hist["L_MO"].append(L_MO)
        hist["U"].append(U)
        hist["solar"].append(solar)

        if n == n_steps:
            break

        # ---- difusao turbulenta (implicita) ----
        Kz_a_prof, h_pbl = Kz_atmosfera(H, tau)
        Kz_a_face = 0.5 * (Kz_a_prof[:-1] + Kz_a_prof[1:])   # media nas interfaces
        hist["h_pbl"].append(h_pbl)

        Kz_o_prof = Kz_oceano(tau)
        Kz_o_face = 0.5 * (Kz_o_prof[:-1] + Kz_o_prof[1:])

        Ta_diff = diffuse_implicit(Ta, Kz_a_face, dt, dz_a)
        qa_diff = diffuse_implicit(qa, Kz_a_face, dt, dz_a)
        To_diff = diffuse_implicit(To, Kz_o_face, dt, dz_o)

        # ---- fontes de superficie (operator splitting, explicito) ----
        Ta_diff[0] += dt * H / (rho_air * cp_air * dz_a)
        qa_diff[0] += dt * LE / (rho_air * L_v * dz_a)

        Q_layers = solar_heating_profile(solar, z_o_edges)
        To_diff += dt * Q_layers / (rho_water * cp_water * dz_o)
        To_diff[0] += dt * (-(H + LE) - LW_net) / (rho_water * cp_water * dz_o)

        Ta, qa, To = Ta_diff, np.clip(qa_diff, 0, None), To_diff

    for key in hist:
        hist[key] = np.array(hist[key])
    return hist


print("Integrando esquema 'bulk simples' (72 h)...")
hist_simples = run_simulation("simples")
print("Integrando esquema 'Monin-Obukhov' (72 h)...")
hist_mo = run_simulation("mo")
print("Concluido.")
