import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# =====================================================================================
# MODELO 2D (x,z) DE ARRASTO POR ONDA DE GRAVIDADE — FONTE OROGRÁFICA + CONVECTIVA
# -------------------------------------------------------------------------------------
# Duas fontes simultâneas, seguindo o "Modelo Simples" do curso (documento
# consolidado, Seção 5; slides "Um Modelo Simples": Orographic Forcing / Convective
# Forcing, com parâmetros a1, a2, Zb, Zt):
#
#   - OROGRÁFICA: montanha tipo Witch-of-Agnesi h(x)=h0·a²/(a²+x²) na superfície
#                 (a1 = 10 km, valor do slide).
#   - CONVECTIVA: "efeito de obstáculo" (documento consolidado, Seção 3.1;
#                 Clark et al., 1986) — o aquecimento da nuvem desloca as
#                 isentrópicas no TOPO da nuvem (z = Zt = 11 km, valor do slide),
#                 que passa a se comportar como uma "montanha efetiva" elevada,
#                 com meia-largura a2 = 5·a1 = 50 km (valor do slide).
#
# FÍSICA DA ONDA 2D (por fonte) — solução clássica de onda de montanha hidrostática
# -------------------------------------------------------------------------------------
# Ponto de partida: a relação de dispersão de ondas de gravidade internas puras do
# curso (Seção 2.2/2.3, Eqs. 7.43–7.45):
#     (ν - Ūk)²(k²+m²) - N²k² = 0                                          (Eq. 7.43)
#     ν̂ = ν - Ūk = ± Nk/κ , κ = √(k²+m²)                                   (Eq. 7.44)
# Para uma onda ESTACIONÁRIA forçada por uma montanha fixa (ν=0, ν̂=-Uk) no limite
# HIDROSTÁTICO (k << m), a relação acima se reduz a m ≈ N/U — o número de onda
# vertical não depende de k, e é essa propriedade que permite escrever a solução
# analítica fechada para a montanha de Agnesi (Queney, 1948):
#     η(x,z) = h0·a·[a·cos(Φ) - x·sin(Φ)] / (a² + x²) ,   Φ(z) = ∫ N/U dz
# — a mesma forma da montanha na superfície, mas "girada" em fase por Φ(z), que é
# a integral vertical acumulada de N/U (generalização WKB para N(z),U(z) variáveis;
# reduz-se a Φ=Nz/U se N,U forem constantes, como nas Eqs. 7.44/7.45). Esse giro de
# fase é o que produz o clássico "tilt para montante" (upstream tilt) das cristas de
# onda com a altura, consequência direta de c_gz ter sinal oposto ao de c_gx na
# Eq. 7.45a/b (propagação de fase para baixo <=> propagação de energia para cima).
#
# A amplitude de pico A(z) (equivalente ao "h0" local) cresce com a diminuição da
# densidade — conservação do fluxo de energia da onda, ~ sqrt(rho_fonte/rho(z)) —
# até ser travada pela hipótese de saturação de Lindzen (documento consolidado,
# Seção 6.1): Ri_w = N_w²/(∂U_w/∂z)² < 1/4 equivale a uma amplitude crítica
#     a_crit(z) = Fc · U(z)/N(z)
# acima da qual a onda quebra. A saturação é aplicada de forma IRREVERSÍVEL (uma
# vez limitada, a envoltória não recupera amplitude em alturas maiores — mesma
# lógica de "mínimo corrente" usada no modelo de coluna 1D para M(z)).
#
# O fluxo de momentum M(z) e o arrasto/aquecimento de coluna (Seção 3.2/3.3/4) são
# obtidos a partir dessa envoltoria de amplitude A(z), e o campo de onda 2D total é
# a SUPERPOSIÇÃO LINEAR das duas fontes (válido porque a teoria é linear).
# =====================================================================================

# ---------- grades ----------
x = np.arange(-150, 150.01, 1.0) * 1000.0        # m
z = np.arange(0, 80.01, 0.25) * 1000.0           # m
nx, nz = len(x), len(z)
dz = z[1] - z[0]

# ---------- atmosfera de fundo (mesma estrutura do modelo de coluna 1D) ----------
H = 7000.0
rho0 = 1.225
rho = rho0 * np.exp(-z / H)

def smooth_step(z, z0, w):
    """Funcao logistica suave (tanh), soh para construir N(z)/U(z) idealizados
    sem descontinuidades -- nao corresponde a nenhuma equacao do curso."""
    return 0.5 * (1 + np.tanh((z - z0) / w))

# N(z): N^2 = (g/theta)(d theta/dz)  -- Eq. 7.4 (Secao 2.1 do documento consolidado)
N = 0.012 + 0.008 * smooth_step(z, 11000, 3000)
# U(z): mesmo jato idealizado do modelo de coluna 1D (troposferico + estratosferico
# + reversao proxima a mesopausa)
U = (10
     + 20 * smooth_step(z, 3000, 3000) * (1 - smooth_step(z, 11000, 3000))
     + 45 * smooth_step(z, 15000, 8000) * (1 - smooth_step(z, 45000, 8000))
     - 90 * smooth_step(z, 60000, 6000))
cp = 1004.0     # J/kg/K

# ---------- fontes (parametros do "Modelo Simples" do curso, Secao 5) ----------
Fc = 1.0                              # fator de saturacao O(1) (analogo a Fr^2/(Fr^2+CG/OC) do GWDO, Secao 6.2.1)
k_mtn = 2 * np.pi / (2 * 10000.0)     # a1 = 10 km (mountain half-width, do slide "Um Modelo Simples")

# x0: posicao horizontal; a: meia-largura (Witch of Agnesi); h0: amplitude da fonte;
# zs: altura da fonte (z=0 para a montanha; z=Zt=11km, topo da nuvem, para a convectiva);
# k: numero de onda horizontal representativo (usado so no calculo do fluxo M, Eq. tipo Secao 6.2.1/6.3.1)
mtn = dict(x0=-70000.0, a=10000.0, h0=250.0, zs=0.0, k=k_mtn)                     # Orographic Forcing (Secao 5)
conv = dict(x0=70000.0, a=50000.0, h0=300.0, zs=11000.0, k=2*np.pi/(2*50000.0))   # Convective Forcing, Zt=11km (Secao 5)

def shape(xx, a):
    """Perfil de Witch of Agnesi normalizado (pico=1) -- usado apenas para desenhar
    o contorno da montanha na figura, nao no calculo da onda (ver source_fields)."""
    return a**2 / (a**2 + xx**2)

def source_fields(src):
    """Para uma fonte (orografica ou convectiva), calcula:
      A(z)   - envoltoria de amplitude de pico do deslocamento vertical (m)
      eta    - campo 2D de deslocamento de isentropica/streamline (x,z), em metros
      tau    - fluxo de momentum COM SINAL, ja pronto para dudt = -(1/rho) dtau/dz
      iz0    - indice da altura da fonte (0 p/ montanha; topo da nuvem p/ convectiva)

    Passo a passo (ver cabecalho do arquivo para a fisica completa):
      (1) Φ(z) = ∫_{zs}^{z} N(z')/U(z') dz'   -- fase vertical acumulada, WKB do
          resultado hidrostatico m=N/U (Eq. 7.44 no limite k<<m); eh o que produz o
          "tilt para montante" da onda com a altura.
      (2) env(z) = h0 * sqrt(rho_fonte/rho(z))  -- crescimento de amplitude por
          conservacao do fluxo de energia da onda (densidade decrescente).
      (3) a_crit(z) = Fc*|U(z)|/N(z)  -- amplitude critica de saturacao de Lindzen
          (Ri_w<1/4, Secao 6.1); A(z) = min(env(z), menor a_crit ja encontrado
          entre zs e z) => saturacao IRREVERSIVEL, identica em espirito ao M(z) do
          modelo de coluna 1D.
      (4) eta(x,z) = A(z)*a*[a*cos(Φ) - (x-x0)*sin(Φ)] / (a² + (x-x0)²)
          -- solucao fechada de Queney (1948) para a montanha de Agnesi, com A(z) e
          Φ(z) generalizados para N(z),U(z) variaveis (aproximacao WKB).
      (5) M(z) = 0.5*rho*N*|U|*A(z)²*k  -- fluxo de momentum de uma onda hidrostatica
          monocromatica de amplitude A(z) (mesma estrutura ~ rho*N*U*(amplitude)^2
          de tau_GWD/GWDC, Secoes 6.2.1/6.3.1); tau = -sign(U_fonte)*M (arrasto
          sempre opoe-se ao vento que gerou a onda).
    """
    iz0 = np.argmin(np.abs(z - src["zs"]))
    A = np.zeros(nz)
    phi = np.zeros(nz)                                            # fase acumulada (rad) -- Φ(z), passo (1)
    Ueff = np.where(np.abs(U) < 3.0, np.sign(U) * 3.0 + (U == 0) * 3.0, U)  # evita divisao por ~0 na integral de fase

    running_cap = np.inf
    z_src = z[iz0]
    rho_src = rho[iz0]
    for i in range(iz0, nz):
        env = src["h0"] * np.sqrt(rho_src / rho[i])               # passo (2): crescimento por densidade
        a_crit = Fc * abs(U[i]) / max(N[i], 1e-6)                 # passo (3): amplitude critica de Lindzen (Secao 6.1)
        running_cap = min(running_cap, a_crit)
        A[i] = min(env, running_cap)                              # saturacao irreversivel
        if i > iz0:
            phi[i] = phi[i-1] + N[i] / Ueff[i] * dz                # Φ(z) = ∫ N/U dz  (WKB da Eq. 7.44)

    a = src["a"]
    dx = x[None, :] - src["x0"]
    eta = np.zeros((nz, nx))
    for i in range(iz0, nz):
        # passo (4): solucao de Queney (1948) para onda de montanha hidrostatica,
        # com amplitude A(z) e fase Φ(z) dependentes da altura (WKB)
        eta[i, :] = A[i] * a * (a * np.cos(phi[i]) - dx[0] * np.sin(phi[i])) / (a**2 + dx[0]**2)

    M = 0.5 * rho * N * np.abs(U) * A**2 * src["k"]               # passo (5): fluxo de momentum (~ rho*N*U*A^2)
    sign0 = np.sign(U[iz0]) if U[iz0] != 0 else 1.0
    tau = np.where(np.arange(nz) >= iz0, -sign0 * M, 0.0)         # estresse com sinal, oposto ao vento na fonte
    return A, eta, tau, iz0

A_mtn, eta_mtn, tau_mtn, iz_mtn = source_fields(mtn)
A_conv, eta_conv, tau_conv, iz_conv = source_fields(conv)

eta_total = eta_mtn + eta_conv                                   # superposicao linear (teoria linear -> valido)

# ---------- arrasto e aquecimento (coluna combinada, Secao 4) ----------
def smooth(v, win=25):
    """Suavizacao com padding por borda -- representa uma camada fisica de
    quebra (nao ruido de grade) e evita artefato espurio nas bordas do dominio."""
    kernel = np.ones(win) / win
    pad = win // 2
    return np.convolve(np.pad(v, pad, mode="edge"), kernel, mode="valid")

tau_mtn_s = smooth(tau_mtn)
tau_conv_s = smooth(tau_conv)
tau_total = tau_mtn_s + tau_conv_s

def drag_from_tau(tau):
    """du/dt = -(1/rho) d(tau)/dz  -- termo 'gwdd' da equacao de Reynolds do
    momentum (documento consolidado, Secao 4)."""
    d = np.zeros(nz)
    d[1:-1] = -(1.0 / rho[1:-1]) * (tau[2:] - tau[:-2]) / (2 * dz)
    d[0], d[-1] = d[1], d[-2]
    return np.clip(d, -6e-4, 6e-4)               # limitador numerico (~52 m/s/dia), soh p/ estabilidade

dudt_mtn = drag_from_tau(tau_mtn_s)
dudt_conv = drag_from_tau(tau_conv_s)
dudt_total = dudt_mtn + dudt_conv
dTdt_total = -(U / cp) * dudt_total              # aquecimento por dissipacao de energia cinetica (extensao ilustrativa)

z_km, x_km = z / 1000.0, x / 1000.0

# =====================================================================
# FIGURA
# =====================================================================
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(1, 4, width_ratios=[3.2, 3.2, 1, 1], wspace=0.35)

# --- painel 1: campo de onda eta(x,z) ---
ax1 = fig.add_subplot(gs[0, 0])
lvl = np.linspace(-350, 350, 29)
cf = ax1.contourf(x_km, z_km, eta_total, levels=lvl, cmap="RdBu_r", extend="both")
ax1.contour(x_km, z_km, eta_total, levels=[0], colors="k", linewidths=0.4, alpha=0.4)
# regiao de quebra: onde a amplitude bateu no teto de saturacao
break_mtn = (A_mtn < mtn["h0"] * np.sqrt(rho[iz_mtn]/rho) - 1e-6) & (np.arange(nz) > iz_mtn)
break_conv = (A_conv < conv["h0"] * np.sqrt(rho[iz_conv]/rho) - 1e-6) & (np.arange(nz) > iz_conv)
for i in range(nz):
    if break_mtn[i]:
        ax1.axhspan(z_km[i]-0.001, z_km[i]+0.13, xmin=0, xmax=1, color="none")
ax1.fill_between(x_km, z_km[break_mtn][0] if break_mtn.any() else 999,
                  z_km[break_mtn][-1] if break_mtn.any() else 0,
                  color="gold", alpha=0.12, zorder=0)
if break_conv.any():
    ax1.fill_between(x_km, z_km[break_conv][0], z_km[break_conv][-1], color="orangered", alpha=0.10, zorder=0)
# montanha
h_mtn_profile = mtn["h0"]/1000 * 8 * shape(x - mtn["x0"], mtn["a"])
ax1.fill_between(x_km, 0, h_mtn_profile, color="#4b3621", zorder=5)
# nuvem convectiva (simbolo)
for dx_, dz_, r in [(-8,0,14),(0,4,18),(9,0,15),(-3,9,12),(5,9,11)]:
    ax1.add_patch(Ellipse((70+dx_, 6+dz_), r, r*0.9, color="#666666", alpha=0.55, zorder=5))
ax1.axhline(conv["zs"]/1000, color="orangered", ls=":", lw=1)
ax1.text(70, 14.5, "topo da nuvem\n(fonte convectiva)", color="orangered", fontsize=7, ha="center")
ax1.text(-70, 58, "quebra / saturação\n(Ri < 1/4)", color="#7a5200", fontsize=7.5, ha="center", weight="bold")
ax1.set_xlabel("x (km)"); ax1.set_ylabel("Altura (km)")
ax1.set_title("(a) Campo de onda η(x,z): fonte + propagação\n(tilt p/ montante, quebra sombreada)")
ax1.set_ylim(0, 80); ax1.set_xlim(-150, 150)
fig.colorbar(cf, ax=ax1, shrink=0.8, label="deslocamento de isentrópica (m)")

# --- painel 2: fluxo de momentum |tau(x,z)| aproximado (produto amplitude x shape) ---
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
flux2d = np.abs(eta_mtn) * 0
for i in range(nz):
    flux2d[i, :] = np.abs(eta_mtn[i, :]) * abs(tau_mtn_s[i]) / (A_mtn[i] + 1e-9) + \
                    np.abs(eta_conv[i, :]) * abs(tau_conv_s[i]) / (A_conv[i] + 1e-9)
cf2 = ax2.contourf(x_km, z_km, flux2d, levels=20, cmap="viridis")
ax2.fill_between(x_km, 0, h_mtn_profile, color="#4b3621", zorder=5)
for dx_, dz_, r in [(-8,0,14),(0,4,18),(9,0,15),(-3,9,12),(5,9,11)]:
    ax2.add_patch(Ellipse((70+dx_, 6+dz_), r, r*0.9, color="#dddddd", alpha=0.6, zorder=5))
ax2.set_xlabel("x (km)")
ax2.set_title("(b) |Fluxo de momentum| local\n(propagação → concentração na quebra)")
ax2.set_xlim(-150, 150)
fig.colorbar(cf2, ax=ax2, shrink=0.8, label="~ Pa")

# --- painel 3: tendencia de vento (coluna combinada) ---
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
ax3.plot(dudt_mtn*86400, z_km, color="#8B5A2B", lw=1.8, label="orográfica")
ax3.plot(dudt_conv*86400, z_km, color="#B22222", lw=1.8, label="convectiva")
ax3.plot(dudt_total*86400, z_km, color="k", lw=2.2, label="total")
ax3.axvline(0, color="gray", lw=0.6)
ax3.set_xlabel("∂u/∂t\n(m/s/dia)")
ax3.set_title("(c) Arrasto\n(coluna)")
ax3.legend(fontsize=7, loc="upper left")

# --- painel 4: tendencia de temperatura ---
ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)
ax4.plot(dTdt_total*86400, z_km, color="k", lw=2.2)
ax4.axvline(0, color="gray", lw=0.6)
ax4.set_xlabel("∂T/∂t\n(K/dia)")
ax4.set_title("(d) Aquecimento\n(coluna)")

for ax in [ax1, ax3, ax4]:
    ax.axhspan(50, 80, color="orange", alpha=0.06)
    ax.axhspan(11, 50, color="steelblue", alpha=0.04)

fig.suptitle("Modelo 2D (x–z) de GWD: fonte orográfica + convectiva, propagação e quebra",
             fontsize=13, y=0.99)
plt.savefig("/home/claude/doc/gwd_2d_model.png", dpi=140, bbox_inches="tight")
print("saved")

i50 = np.argmin(np.abs(z_km-50)); i40 = np.argmin(np.abs(z_km-40)); i30=np.argmin(np.abs(z_km-30))
print(f"drag total em 30/40/50 km (m/s/dia): {dudt_total[i30]*86400:.1f} / {dudt_total[i40]*86400:.1f} / {dudt_total[i50]*86400:.1f}")
print(f"dT/dt total em 30/40/50 km (K/dia):  {dTdt_total[i30]*86400:.2f} / {dTdt_total[i40]*86400:.2f} / {dTdt_total[i50]*86400:.2f}")
