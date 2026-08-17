import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================================
# MODELO DE COLUNA 1D DE ARRASTO POR ONDA DE GRAVIDADE OROGRÁFICA (GWDO)
# -------------------------------------------------------------------------------------
# Reproduz numericamente o ciclo Nascimento -> Evolução -> Dissipação descrito no
# documento consolidado (Seções 3, 4 e 6) e no material da disciplina MET-576-4
# (Prof. Paulo Kubota):
#   "Oscilações Atmosféricas — Teoria da Perturbação Linear, parte 1" (equações 7.x)
#   "Gravity Wave Drag Parameterization" (esquemas GWDO/GWDC)
#
# CORRESPONDÊNCIA CÓDIGO <-> EQUAÇÕES DO CURSO
# -------------------------------------------------------------------------------------
#  Etapa           Variável no código      Equação / conceito no material do curso
#  --------------  ----------------------  --------------------------------------------
#  Estabilidade    N(z)                    N² = (g/θ)(∂θ/∂z)                    (Eq. 7.4)
#  Fonte           M0 = τ0                 τ_GWD ∝ ρ0·N0·U0·h²·k  (forma linear de
#                                           montanha; mesma estrutura de
#                                           τ_GWD = -E(m/Δx)(ρ0U0³/N0)Fr²/(Fr²+CG/OC),
#                                           Seção 6.2.1 / slide "GWDO")
#  Propagação      M(z) conservado         dM/dz = 0 sem dissipação (Seção 3.2);
#                                           M(z) = ρ0 ∫ u'w' dx
#  Dissipação      M_sat(z), saturação     Hipótese de saturação de Lindzen (1981):
#                                           Ri_w = N_w²/(∂U_w/∂z)² < 1/4  (Seção 6.1)
#                                           -> M_sat(z) = Fc·ρ(z)|U(z)|³k/N(z)
#                                           (mesma estrutura ρU³/N de τ_GWD e de
#                                           GWDC = -(ρ0U0³/ΔxN0)c1c2²τ̂0², Seção 6.3.1)
#  Arrasto         dudt                    ∂u/∂t = -(1/ρ)·∂τ/∂z   (termo "gwdd" da
#                                           equação de Reynolds, Seção 4)
#  Aquecimento     dTdt                    Extensão ilustrativa: dT/dt = -(U/cp)·∂u/∂t
#                                           (conversão de energia cinética dissipada em
#                                           calor; o curso apenas indica que um termo
#                                           GWDD aparece na eq. de calor, sem fórmula
#                                           fechada — ver nota no documento consolidado)
#
# HIPÓTESES SIMPLIFICADORAS (coerentes com o "Modelo Simples" da Seção 5):
#   - onda monocromática (um único par k0/h representando a orografia de sub-grade),
#     hidrostática, linear, sem rotação, Boussinesq no cálculo da amplitude;
#   - só a MAGNITUDE do fluxo de momentum é propagada (não a fase), o que é suficiente
#     para obter perfis de arrasto/aquecimento em função da altura;
#   - nível crítico (U->0) é tratado de forma suave: M_sat->0 quando |U|->0, absorvendo
#     a onda sem singularidade numérica (equivale ao "nível crítico de filtragem" da
#     Seção 3.2/3.3, sem precisar de um teste explícito de troca de sinal de U).
# =====================================================================================

# ---------- grade vertical ----------
z = np.arange(0, 80.01, 0.25) * 1000.0     # m, superficie ateh 80 km
dz = z[1] - z[0]
nz = len(z)

# ---------- atmosfera de fundo (idealizada, inverno, Hem. Norte) ----------
H = 7000.0                                  # altura de escala, m  (rho = rho0 exp(-z/H))
rho0 = 1.225
rho = rho0 * np.exp(-z / H)

def smooth_step(z, z0, w):
    """Funcao logistica suave (tanh) usada soh para construir perfis idealizados
    de N(z) e U(z) sem descontinuidades - nao corresponde a uma equacao do curso,
    eh apenas uma conveniencia numerica."""
    return 0.5 * (1 + np.tanh((z - z0) / w))

# N(z): frequencia de Brunt-Vaisala, N^2 = (g/theta)(d theta/dz)  -- Eq. 7.4 (Secao 2.1)
# troposfera ~0.012 s^-1, estratosfera/mesosfera mais estavel ~0.020 s^-1
N = 0.012 + 0.008 * smooth_step(z, 11000, 3000)

# U(z): vento de fundo idealizado -- jato subtropical na tropopausa + jato
# estratosferico de inverno crescendo com a altura + reversao proxima a mesopausa
# (regiao classica de deposicao de GWD / "nivel critico de filtragem", Secao 3.2/3.3)
U = (10
     + 20 * smooth_step(z, 3000, 3000) * (1 - smooth_step(z, 11000, 3000))   # jato troposferico
     + 45 * smooth_step(z, 15000, 8000) * (1 - smooth_step(z, 45000, 8000))  # jato estratosferico
     - 90 * smooth_step(z, 60000, 6000))                                     # reversao proximo ao topo

cp = 1004.0   # J/kg/K, calor especifico do ar seco a pressao constante

# ---------- fonte orografica (onda de montanha, teoria linear hidrostatica) ----------
h_mtn = 250.0                      # m, altura rms da orografia de sub-grade ("h" da Secao 6.1/6.2)
lambda0 = 8000.0                   # m, comprimento de onda horizontal dominante da montanha
k0 = 2 * np.pi / lambda0           # numero de onda horizontal correspondente
Fc = 1.0                           # fator de saturacao O(1), analogo a Fr^2/(Fr^2+CG/OC) do GWDO (Eq. Secao 6.2.1)

sign0 = np.sign(U[0]) if U[0] != 0 else 1.0
# M0 = |tau0|: fluxo de momentum na fonte, forma classica de onda de montanha linear
# hidrostatica (rho*N*U*h^2*k), com a MESMA estrutura fisica de tau_GWD (Secao 6.2.1)
# e de GWDC (Secao 6.3.1): sempre proporcional a rho*N*U*(amplitude)^2.
M0 = rho[0] * N[0] * abs(U[0]) * h_mtn**2 * k0     # |fluxo de momentum| na fonte (Pa)

def compute_flux_profile(U, N, rho, M0):
    """Propaga a MAGNITUDE do fluxo de momentum M(z) a partir da fonte.

    Fisica (documento consolidado, Secoes 3.2/3.3 e 6.1):
      1) EVOLUCAO / propagacao: na ausencia de dissipacao, M(z) se conserva com a
         altura -- dM/dz = 0 (Secao 3.2), isto eh, M(z) = M(z-dz).
      2) DISSIPACAO / quebra: a hipotese de saturacao de Lindzen (1981) afirma que a
         onda quebra quando o numero de Richardson da onda cai abaixo de 1/4:
             Ri_w = N_w^2 / (dU_w/dz)^2  <  1/4                      (Secao 6.1)
         o que equivale a limitar o fluxo sustentavel em cada nivel a um valor de
         saturacao M_sat(z). Usamos a forma classica (McFarlane, 1987; mesma
         estrutura de tau_GWD e GWDC do curso, ambas ~ rho*U^3/N):
             M_sat(z) = Fc * rho(z) * |U(z)|^3 * k0 / N(z)
      3) O fluxo real e o MINIMO entre o que "sobrou" do nivel de baixo e o que o
         nivel atual consegue sustentar -- garante que a quebra seja irreversivel
         (a onda nao recupera amplitude acima de onde ja quebrou):
             M(z) = min( M(z-dz), M_sat(z) )
      4) Proximo a um nivel critico (U(z) -> 0), M_sat -> 0 suavemente, absorvendo
         a onda sem singularidade numerica -- equivalente ao "nivel critico de
         filtragem" citado na Secao 3.2/3.3, sem precisar testar explicitamente a
         troca de sinal de U.

    O estresse com sinal e definido opondo-se ao vento na fonte (tau = -sign(U0)*M),
    consistente com o arrasto (a onda sempre remove momentum na direcao do
    escoamento que a gerou -- e por isso se chama "drag").
    """
    M = np.zeros(nz)
    M[0] = M0
    for i in range(1, nz):
        M_sat = Fc * rho[i] * abs(U[i])**3 * k0 / max(N[i], 1e-5)   # saturacao de Lindzen, Secao 6.1
        M[i] = min(M[i-1], M_sat)                                    # conservacao (Secao 3.2) + quebra irreversivel
    # suavizacao (representa uma camada fisica de quebra, nao ruido de grade)
    # -- padding por repeticao da borda, para nao criar artefato espurio em z=0/z=80km
    win = 33
    kernel = np.ones(win) / win
    pad = win // 2
    M_padded = np.pad(M, pad, mode="edge")
    M = np.convolve(M_padded, kernel, mode="valid")
    tau_signed = -sign0 * M            # estresse com sinal (opoe-se ao vento na fonte)
    return M, tau_signed

def tendencies(U, N, rho, M0):
    """Calcula as tendencias de vento e temperatura a partir da divergencia do
    fluxo de momentum (arrasto) -- Secao 4 do documento consolidado.

    Arrasto (mesmo termo "gwdd" que aparece na equacao de Reynolds do momentum,
    Secao 4):
        du/dt = -(1/rho) * d(tau)/dz

    Aquecimento associado a dissipacao da energia cinetica da onda (extensao
    ilustrativa proposta para complementar o curso -- o material original apenas
    menciona que um termo GWDD aparece, de forma analoga, na equacao de calor,
    sem fornecer uma formula fechada; aqui aproximamos como conversao local de
    energia cinetica em calor):
        dT/dt = -(U/cp) * du/dt
    """
    M, tau = compute_flux_profile(U, N, rho, M0)
    dudt = np.zeros(nz)
    dudt[1:-1] = -(1.0 / rho[1:-1]) * (tau[2:] - tau[:-2]) / (2 * dz)   # du/dt = -(1/rho) dtau/dz  (Secao 4)
    dudt[0] = dudt[1]
    dudt[-1] = dudt[-2]
    dudt = np.clip(dudt, -6e-4, 6e-4)          # limitador numerico (~52 m/s/dia, so p/ estabilidade)
    dTdt = -(U / cp) * dudt                    # aquecimento por dissipacao de energia cinetica (extensao ilustrativa)
    return tau, dudt, dTdt

tau, dudt, dTdt = tendencies(U, N, rho, M0)

# ---------- impacto acumulado em N dias (extrapolacao com forcante congelada) ----------
# Para evitar realimentacao instavel em um modelo de coluna tao idealizado,
# o impacto acumulado eh estimado extrapolando a tendencia inicial no tempo,
# limitando a mudanca a no maximo 90% do vento original em cada nivel
# (mesmo espirito do limitador de 50% do CB98/CB02, Secao 6.3.2).
days = 5
delta = dudt * (days * 86400.0)
max_change = 0.9 * np.abs(U)
delta = np.clip(delta, -max_change, max_change)
U_eq = U + delta
tau_final, dudt_final, dTdt_final = tendencies(U_eq, N, rho, M0)
tau0 = M0

z_km = z / 1000.0

# =====================================================================
# FIGURA
# =====================================================================
fig, axs = plt.subplots(1, 5, figsize=(19, 8), sharey=True)
fig.suptitle("Modelo de Coluna 1D de GWD Orográfico — Fonte, Propagação e Dissipação\n"
             "(equações do curso MET-576-4: saturação de Lindzen, τ ∝ ρU³/N)", fontsize=13, y=1.02)

axs[0].plot(U, z_km, color="#1F3864", lw=2, label="U inicial")
axs[0].plot(U_eq, z_km, color="#C0392B", lw=2, ls="--", label=f"U após {days} dias c/ GWD")
axs[0].axvline(0, color="gray", lw=0.7)
axs[0].set_xlabel("Vento zonal U (m/s)")
axs[0].set_ylabel("Altura (km)")
axs[0].set_title("(a) Vento de fundo")
axs[0].legend(fontsize=8, loc="upper left")
axs[0].grid(alpha=0.3)

axs[1].plot(N * 1000, z_km, color="#2E7D32", lw=2)
axs[1].set_xlabel("N (×10⁻³ s⁻¹)")
axs[1].set_title("(b) Estabilidade (Brunt-Väisälä)")
axs[1].grid(alpha=0.3)

axs[2].plot(tau / tau0, z_km, color="#1F3864", lw=2, label="τ(z)/τ₀ (t=0)")
axs[2].plot(tau_final / tau0, z_km, color="#C0392B", lw=2, ls="--", label=f"τ(z)/τ₀ (t={days}d)")
axs[2].axvline(0, color="gray", lw=0.7)
axs[2].set_xlabel("Fluxo de momentum normalizado")
axs[2].set_title("(c) Propagação: M(z)\nconservado até a quebra")
axs[2].legend(fontsize=8)
axs[2].grid(alpha=0.3)

axs[3].plot(dudt * 86400, z_km, color="#1F3864", lw=2, label="t=0")
axs[3].plot(dudt_final * 86400, z_km, color="#C0392B", lw=2, ls="--", label=f"t={days}d")
axs[3].axvline(0, color="gray", lw=0.7)
axs[3].set_xlabel("∂u/∂t (m/s por dia)")
axs[3].set_title("(d) Arrasto: tendência\nde momento")
axs[3].legend(fontsize=8)
axs[3].grid(alpha=0.3)

axs[4].plot(dTdt * 86400, z_km, color="#1F3864", lw=2, label="t=0")
axs[4].plot(dTdt_final * 86400, z_km, color="#C0392B", lw=2, ls="--", label=f"t={days}d")
axs[4].axvline(0, color="gray", lw=0.7)
axs[4].set_xlabel("∂T/∂t (K por dia)")
axs[4].set_title("(e) Aquecimento associado\nà dissipação")
axs[4].legend(fontsize=8)
axs[4].grid(alpha=0.3)

for ax in axs:
    ax.set_ylim(0, 80)
    ax.axhspan(50, 80, color="orange", alpha=0.08)
    ax.axhspan(11, 50, color="steelblue", alpha=0.06)

axs[0].text(-85, 65, "Mesosfera\n(nível crítico /\nquebra intensa)", fontsize=8, color="#8a5a00")
axs[0].text(-85, 30, "Estratosfera", fontsize=8, color="#2b4f77")

plt.tight_layout()
plt.savefig("/home/claude/doc/gwd_column_model.png", dpi=140, bbox_inches="tight")
print("saved")

# ---------- diagnósticos numéricos de alto nível ----------
i50 = np.argmin(np.abs(z_km - 50))
i65 = np.argmin(np.abs(z_km - 65))
i70 = np.argmin(np.abs(z_km - 70))
print(f"tau0 (fonte, superficie)      = {tau0:.4f} Pa")
print(f"U em 50 km  (t=0 / t={days}d)  = {U[i50]:.1f} / {U_eq[i50]:.1f} m/s")
print(f"U em 65 km  (t=0 / t={days}d)  = {U[i65]:.1f} / {U_eq[i65]:.1f} m/s")
print(f"U em 70 km  (t=0 / t={days}d)  = {U[i70]:.1f} / {U_eq[i70]:.1f} m/s")
print(f"dudt max (m/s/dia) em t=0     = {np.min(dudt*86400):.2f}  na altura {z_km[np.argmin(dudt)]:.1f} km")
print(f"dTdt max (K/dia) em t=0       = {np.max(np.abs(dTdt*86400)):.3f} na altura {z_km[np.argmax(np.abs(dTdt))]:.1f} km")
