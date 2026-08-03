# -*- coding: utf-8 -*-
"""
================================================================================
Grupo 3 - MET-579 -- MODELO PROGNOSTICO FUNCIONAL (v2: bolha OU montanha)
Integracao no tempo das equacoes 2.1-2.4 do relatorio (mais vento basico U),
com duas opcoes de condicao inicial/contorno sinteticas, selecionaveis por
MODO logo abaixo:

  MODO = "bolha"    -> bolha de empuxo isolada (mesma CI da v1 deste script),
                       dominio em repouso, sem vento basico, tampa rigida
                       (psi'=0) no topo e na base.
  MODO = "montanha" -> escoamento com vento basico U sobre uma cordilheira
                       senoidal (fonte topografica, tipo "ondas de montanha"
                       de Holton, Sec. 7.4.2), com camada esponja no topo
                       para absorver a energia irradiada (condicao de
                       radiacao sintetica) em vez de refletir na tampa
                       rigida.

Ambos os modos resolvem exatamente o mesmo sistema fisico (reformulacao em
vorticidade-funcao de corrente das Eqs. 2.1-2.4, ver docstring da v1) -- o
que muda e a condicao inicial, a condicao de contorno inferior e a presenca
(ou nao) de vento basico e camada esponja.

--------------------------------------------------------------------------------
REFORMULACAO (recapitulando a v1)
--------------------------------------------------------------------------------
Com vento basico U constante em x, o sistema (2.1)-(2.4) linearizado em
torno do escoamento U vira, apos a mesma eliminacao de pressao:

    (V1)  dzeta'/dt = -U dzeta'/dx + db'/dx
    (V2)  d2psi'/dx2 + d2psi'/dz2 = zeta'
    (V3)  db'/dt   = -U db'/dx   - N^2 dpsi'/dx

que se reduz ao sistema da v1 quando U=0.

--------------------------------------------------------------------------------
CONDICAO TOPOGRAFICA (MODO="montanha")
--------------------------------------------------------------------------------
Uma cordilheira senoidal h(x) = hM*cos(k_topo x) forca o escoamento por
baixo. A condicao cinematica linearizada (Holton, Sec. 7.4.2) exige que a
velocidade vertical na superficie acompanhe a inclinacao do terreno sendo
advectada pelo vento basico,

    w'(x, 0) ~= U dh/dx  ,

e como w' = dpsi'/dx, integrando em x obtem-se diretamente a condicao de
contorno em termos da funcao de corrente:

    psi'(x, 0) = U h(x) .

Esta e a condicao de contorno INFERIOR sintetica deste modo (substitui o
psi'=0 do modo "bolha"). A forcante e ligada suavemente ao longo de
T_RAMPA (evitando um choque inicial que excitaria transientes espurios).
No topo mantem-se psi'=0 (tampa rigida), mas com uma CAMADA ESPONJA
(amortecimento de Rayleigh) nas ultimas ~30% do dominio vertical para
absorver a energia da onda antes que ela reflita na tampa -- uma condicao
de radiacao sintetica simplificada, tecnica padrao em modelos de mesoescala.

Este cenario e projetado para ficar no regime PROPAGANTE (nao evanescente):
com U e k_topo escolhidos de forma que |U k_topo| < N (Holton, Eq. 7.47),
a onda se propaga verticalmente a partir da cordilheira, em vez de ficar
presa (evanescente) junto ao terreno.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================
# 0) ESCOLHA DO MODO -- troque aqui para alternar entre os dois cenarios
# ==================================================================
MODO = "montanha"     # "bolha" ou "montanha"

# ------------------------------------------------------------------
# 1) Parametros fisicos e numericos (dependem do modo escolhido)
# ------------------------------------------------------------------
N_BV = 0.012          # frequencia de Brunt-Vaisala [s^-1] (comum aos 2 modos)

if MODO == "bolha":
    Lx, Lz = 8000.0, 8000.0
    Nx, Nz = 128, 129
    dt = 5.0
    T_total = 2400.0
    tempos_salvos = [0.0, 600.0, 1200.0, 1800.0, 2400.0]
    U_VENTO = 0.0
    T_RAMPA = 0.0
    Z_ESPONJA_FRAC = None   # sem esponja no modo bolha

    b0 = 0.05
    sigma_x = 300.0
    sigma_z = 300.0

elif MODO == "montanha":
    Lx, Lz = 5000.0, 12000.0     # 1 cordilheira preenche todo o dominio periodico
    Nx, Nz = 64, 81
    dt = 8.0
    T_total = 14400.0             # 4 h, tempo suficiente p/ o regime quase-estacionario se estabelecer
    tempos_salvos = [0.0, 3600.0, 7200.0, 10800.0, 14400.0]
    U_VENTO = 5.0                 # vento basico [m/s]
    T_RAMPA = 3600.0              # 1 h para ligar a forcante suavemente
    Z_ESPONJA_FRAC = 0.7           # esponja nos 30% superiores do dominio
    MU_MAX = 1.0 / 600.0           # taxa maxima de amortecimento da esponja [s^-1]

    hM = 100.0                    # altura da cordilheira [m]
    n_cristas = 1                 # numero de cristas ao longo de Lx

else:
    raise ValueError("MODO deve ser 'bolha' ou 'montanha'")

dx = Lx / Nx
dz = Lz / (Nz - 1)
x = np.arange(Nx) * dx
z = np.linspace(0.0, Lz, Nz)
n_steps = int(T_total / dt)

print(f"MODO = {MODO}")
print(f"Grade: Nx={Nx} (dx={dx:.1f} m), Nz={Nz} (dz={dz:.1f} m)")
print(f"dt={dt:.1f} s")

# ------------------------------------------------------------------
# 2) Solver tridiagonal em lote (algoritmo de Thomas vetorizado em k)
# ------------------------------------------------------------------
def thomas_solve_batch(a, b, c, d):
    """Resolve, para cada linha (cada numero de onda k), o sistema
    tridiagonal a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i.
    a, b, c, d tem shape (Nk, Nint); retorna x com a mesma shape."""
    Nk, Nint = d.shape
    cp = np.zeros((Nk, Nint), dtype=complex)
    dp = np.zeros((Nk, Nint), dtype=complex)
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / b[:, 0]
    for i in range(1, Nint):
        denom = b[:, i] - a[:, i] * cp[:, i - 1]
        cp[:, i] = c[:, i] / denom
        dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / denom
    xsol = np.zeros((Nk, Nint), dtype=complex)
    xsol[:, -1] = dp[:, -1]
    for i in range(Nint - 2, -1, -1):
        xsol[:, i] = dp[:, i] - cp[:, i] * xsol[:, i + 1]
    return xsol


# ------------------------------------------------------------------
# 3) Solve eliptico (vetorizado em k): psi_hat(k,z) a partir de
#    zeta_hat(k,z), com contornos psi_bottom_hat(k) e psi_top_hat(k)
#    (ambos default =0, isto e, tampa rigida nas duas bordas)
# ------------------------------------------------------------------
Nint = Nz - 2


def resolver_psi_todos_k(zeta_hat, k_arr, psi_bottom_hat=None, psi_top_hat=None):
    """zeta_hat: shape (Nk, Nz). Retorna psi_hat: shape (Nk, Nz)."""
    Nk = zeta_hat.shape[0]
    if psi_bottom_hat is None:
        psi_bottom_hat = np.zeros(Nk, dtype=complex)
    if psi_top_hat is None:
        psi_top_hat = np.zeros(Nk, dtype=complex)

    a = np.full((Nk, Nint), 1.0 / dz**2, dtype=complex)
    c = np.full((Nk, Nint), 1.0 / dz**2, dtype=complex)
    b = -2.0 / dz**2 - k_arr[:, None] ** 2
    b = np.broadcast_to(b, (Nk, Nint)).copy()

    d = zeta_hat[:, 1:-1].copy()
    d[:, 0] -= psi_bottom_hat / dz**2
    d[:, -1] -= psi_top_hat / dz**2

    psi_int = thomas_solve_batch(a, b, c, d)
    psi_hat = np.zeros((Nk, Nz), dtype=complex)
    psi_hat[:, 0] = psi_bottom_hat
    psi_hat[:, -1] = psi_top_hat
    psi_hat[:, 1:-1] = psi_int
    return psi_hat


# ------------------------------------------------------------------
# 4) Numeros de onda horizontais e camada esponja (se aplicavel)
# ------------------------------------------------------------------
k_wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)
Nk = len(k_wavenumbers)

if MODO == "montanha":
    z_esponja_ini = Z_ESPONJA_FRAC * Lz
    mu_z = np.where(
        z > z_esponja_ini,
        MU_MAX * ((z - z_esponja_ini) / (Lz - z_esponja_ini)) ** 2,
        0.0,
    )
else:
    mu_z = np.zeros(Nz)


def rampa(t):
    """Liga a forcante topografica suavemente ao longo de T_RAMPA (0->1)."""
    if T_RAMPA <= 0:
        return 1.0
    if t >= T_RAMPA:
        return 1.0
    return 0.5 * (1.0 - np.cos(np.pi * t / T_RAMPA))


# ------------------------------------------------------------------
# 5) Forcante topografica em espaco espectral (so usada no modo montanha)
# ------------------------------------------------------------------
if MODO == "montanha":
    k_topo = 2.0 * np.pi * n_cristas / Lx
    h_fisico = hM * np.cos(k_topo * x)
    # psi'(x,0) = U*h(x)  ->  em espectral, psi_bottom_hat(k) = U * rfft(h(x))(k)
    psi_bottom_hat_base = U_VENTO * np.fft.rfft(h_fisico)
    m_topo = np.sqrt(max(N_BV**2 / U_VENTO**2 - k_topo**2, 0.0)) if N_BV / U_VENTO > k_topo else None
    if m_topo:
        print(f"Regime PROPAGANTE previsto: |U k_topo|={U_VENTO*k_topo:.2e} < N={N_BV:.2e}")
        print(f"Comprimento de onda vertical previsto: Lz_onda = {2*np.pi/m_topo:.0f} m")
    else:
        print(f"Regime EVANESCENTE previsto: |U k_topo|={U_VENTO*k_topo:.2e} >= N={N_BV:.2e}")


# ------------------------------------------------------------------
# 6) Derivada temporal do sistema (V1)-(V3), com adveccao U e esponja
# ------------------------------------------------------------------
def tendencias(zeta_hat, b_hat, t):
    ik = 1j * k_wavenumbers[:, None]  # shape (Nk,1), multiplica por broadcasting em z

    if MODO == "montanha":
        psi_bottom_hat = psi_bottom_hat_base * rampa(t)
    else:
        psi_bottom_hat = None

    psi_hat = resolver_psi_todos_k(zeta_hat, k_wavenumbers, psi_bottom_hat=psi_bottom_hat)

    dzeta_dt = -U_VENTO * ik * zeta_hat + ik * b_hat
    db_dt = -U_VENTO * ik * b_hat - N_BV**2 * ik * psi_hat

    # camada esponja (amortecimento de Rayleigh para o estado de repouso)
    dzeta_dt -= mu_z[None, :] * zeta_hat
    db_dt -= mu_z[None, :] * b_hat

    return dzeta_dt, db_dt


def passo_rk4(zeta_hat, b_hat, t, dt):
    k1z, k1b = tendencias(zeta_hat, b_hat, t)
    k2z, k2b = tendencias(zeta_hat + 0.5 * dt * k1z, b_hat + 0.5 * dt * k1b, t + 0.5 * dt)
    k3z, k3b = tendencias(zeta_hat + 0.5 * dt * k2z, b_hat + 0.5 * dt * k2b, t + 0.5 * dt)
    k4z, k4b = tendencias(zeta_hat + dt * k3z, b_hat + dt * k3b, t + dt)
    zeta_novo = zeta_hat + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
    b_novo = b_hat + (dt / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
    return zeta_novo, b_novo


# ------------------------------------------------------------------
# 7) Condicao inicial (depende do modo)
# ------------------------------------------------------------------
X, Z = np.meshgrid(x, z, indexing="ij")

if MODO == "bolha":
    x0, z0 = Lx / 2.0, Lz / 2.0
    b_fisico0 = b0 * np.exp(-((X - x0) ** 2) / sigma_x**2) * np.exp(-((Z - z0) ** 2) / sigma_z**2)
    zeta_fisico0 = np.zeros_like(b_fisico0)
else:  # montanha: comeca em repouso: a forcante topografica e' que gera tudo
    b_fisico0 = np.zeros_like(X)
    zeta_fisico0 = np.zeros_like(X)

b_hat = np.fft.rfft(b_fisico0, axis=0)
zeta_hat = np.fft.rfft(zeta_fisico0, axis=0)

# ------------------------------------------------------------------
# 8) Integracao no tempo, salvando snapshots de b'(x,z)
# ------------------------------------------------------------------
snapshots = {}
proximo_indice_salvar = 0
if 0.0 in tempos_salvos:
    snapshots[0.0] = np.fft.irfft(b_hat, n=Nx, axis=0).copy()
    proximo_indice_salvar = 1

for passo in range(1, n_steps + 1):
    t_antes = (passo - 1) * dt
    zeta_hat, b_hat = passo_rk4(zeta_hat, b_hat, t_antes, dt)
    t_atual = passo * dt
    if proximo_indice_salvar < len(tempos_salvos) and \
       abs(t_atual - tempos_salvos[proximo_indice_salvar]) < dt / 2:
        b_fisico = np.fft.irfft(b_hat, n=Nx, axis=0)
        snapshots[tempos_salvos[proximo_indice_salvar]] = b_fisico.copy()
        print(f"  snapshot salvo em t={t_atual:.0f} s "
              f"(max|b'|={np.max(np.abs(b_fisico)):.4f})")
        proximo_indice_salvar += 1

print("Integracao concluida.")

# ------------------------------------------------------------------
# 9) Figura
# ------------------------------------------------------------------
n_paineis = len(snapshots)
fig, axes = plt.subplots(1, n_paineis, figsize=(4.2 * n_paineis, 5.2), sharey=True)
if n_paineis == 1:
    axes = [axes]

vmax = max(np.max(np.abs(v)) for v in snapshots.values())
vmax = vmax if vmax > 0 else 1.0

for ax, (t_s, campo) in zip(axes, snapshots.items()):
    pcm = ax.pcolormesh(x / 1000.0, z / 1000.0, campo.T, shading="auto",
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    if MODO == "montanha":
        exagero = 20.0
        ax.plot(x / 1000.0, hM * np.cos(k_topo * x) * exagero / 1000.0, color="k", lw=1.2)
        ax.axhline(z_esponja_ini / 1000.0, color="gray", ls=":", lw=1)
    ax.set_title(f"t = {t_s:.0f} s", fontsize=11, fontweight="bold")
    ax.set_xlabel("x [km]")
axes[0].set_ylabel("z [km]")

titulo = ("Modelo prognostico (Eqs. 2.1-2.4 + vento U): ondas de montanha\n"
          "cordilheira senoidal, esponja no topo, RK4"
          if MODO == "montanha" else
          "Modelo prognostico (Eqs. 2.1-2.4): propagacao de bolha de empuxo\n"
          "sistema vorticidade-funcao de corrente, RK4, CC sinteticas de tampa rigida")
fig.suptitle(titulo, fontsize=12, fontweight="bold", y=1.06)
fig.colorbar(pcm, ax=axes, orientation="vertical", shrink=0.8, label="b' [m/s^2]")
nome_fig = "modelo_prognostico_montanha.png" if MODO == "montanha" else "modelo_prognostico_bolha.png"
fig.savefig(nome_fig, dpi=160, bbox_inches="tight")
print(f"Figura salva em {nome_fig}")

# ------------------------------------------------------------------
# 10) Verificacao com a teoria
# ------------------------------------------------------------------
if MODO == "bolha":
    k_dominante = 1.0 / sigma_x
    m_dominante = 1.0 / sigma_z
    alpha_previsto = np.degrees(np.arctan2(k_dominante, m_dominante))
    omega_previsto = N_BV * k_dominante / np.sqrt(k_dominante**2 + m_dominante**2)
    print(f"\nAngulo previsto das linhas de fase (bolha isotropica): ~{alpha_previsto:.0f} graus")
    print(f"Frequencia intrinseca dominante prevista: omega ~ {omega_previsto:.4f} s^-1 "
          f"(periodo ~ {2*np.pi/omega_previsto:.0f} s)")
else:
    if m_topo:
        alpha_previsto = np.degrees(np.arctan2(k_topo, m_topo))
        print(f"\nAngulo previsto das linhas de fase em relacao a vertical: ~{alpha_previsto:.0f} graus")
        print("Fase deve inclinar para OESTE com a altura (energia se propagando para cima),"
              " conforme discutido no Cap. 4 do relatorio (deslocamento Doppler / nivel critico).")
