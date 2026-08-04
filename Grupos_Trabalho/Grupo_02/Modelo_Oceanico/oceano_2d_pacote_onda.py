# -*- coding: utf-8 -*-
"""
MET-576-4 - Extensao 2D: propagacao de um pacote de ondas internas

Demonstracao central da extensao 2D: uma perturbacao de empuxo
LOCALIZADA (envelope gaussiano em x e z) e' decomposta automaticamente
pela FFT em uma superposicao de modos de Fourier, cada um oscilando na
sua propria frequencia omega(kx,kz) -- a assinatura fisica de uma onda
interna de gravidade se propagando e se dispersando no espaco e' a
INTERFERENCIA dessas componentes, algo que o modelo 0D (um unico modo)
nao pode representar.

Comparam-se os tres esquemas (RK4, AB2, Leapfrog) na evolucao desse
pacote por um numero fixo de passos, com:
  - Snapshots do campo de empuxo b(x,z,t) em varios instantes
  - Diagnostico de energia total ao longo do tempo (generalizacao 2D
    do diagnostico ja usado no modelo 0D)

Saidas:
  mo576_2d_pacote_snapshots.png
  mo576_2d_pacote_energia.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oceano_2d_schemes import (
    criar_grade, derivada_espectral, rk4_espectral,
    adams_bashforth2_espectral, inicializar_ab2_espectral,
    leapfrog_espectral, inicializar_leapfrog_espectral,
    robert_asselin_filtro, campos_fisicos, energia_fisica,
)

# ---------------------------------------------------------------------
# Grade e parametros fisicos
# ---------------------------------------------------------------------
nx, nz = 128, 64
Lx, Lz = 200000.0, 2000.0  # 200 km x 2 km
N = 0.01

x, z, dx, dz, KX, KZ, K2 = criar_grade(nx, nz, Lx, Lz)

# dt escolhido para manter N dt pequeno o bastante para o AB2 nao
# divergir de forma dominante no numero de passos desta demonstracao
# (ver discussao de instabilidade fraca do AB2 nos scripts 0D)
dt = 20.0
n_passos = 400
t_fim = n_passos * dt

# ---------------------------------------------------------------------
# Condicao inicial: pacote de empuxo gaussiano localizado no centro do
# dominio, com estrutura vertical mais fina que a horizontal (razoavel
# para uma perturbacao oceanica idealizada)
# ---------------------------------------------------------------------
x0, z0 = Lx / 2.0, Lz / 2.0
sigma_x, sigma_z = 8000.0, 150.0  # m

X2, Z2 = np.meshgrid(x, z)  # forma (nz, nx), mesma convencao da grade espectral
b_inicial = 0.01 * np.exp(-((X2 - x0) ** 2) / (2 * sigma_x ** 2)
                           - ((Z2 - z0) ** 2) / (2 * sigma_z ** 2))
zeta_inicial = np.zeros_like(b_inicial)  # comeca em repouso (sem vorticidade)

b_hat0 = np.fft.fft2(b_inicial)
zeta_hat0 = np.fft.fft2(zeta_inicial)

E0 = energia_fisica(zeta_hat0, b_hat0, KX, KZ, K2, N, dx, dz)
print(f"Energia inicial do pacote: {E0:.6e}")
print(f"dt = {dt:.1f} s, N dt = {N*dt:.3f}, {n_passos} passos, "
      f"t_fim = {t_fim/3600.0:.2f} h")


# ---------------------------------------------------------------------
# Integracao dos 3 esquemas, guardando energia e snapshots
# ---------------------------------------------------------------------
passos_snapshot = [0, 100, 200, 300, 400]


def integrar(nome_esquema):
    energias = np.zeros(n_passos + 1)
    snapshots = {}

    if nome_esquema == "RK4":
        zeta, b = zeta_hat0.copy(), b_hat0.copy()
        energias[0] = energia_fisica(zeta, b, KX, KZ, K2, N, dx, dz)
        u_f, w_f, b_f, zeta_f = campos_fisicos(zeta, b, KX, KZ, K2)
        if 0 in passos_snapshot:
            snapshots[0] = b_f.copy()
        for n in range(n_passos):
            zeta, b = rk4_espectral(zeta, b, KX, K2, N, dt)
            energias[n + 1] = energia_fisica(zeta, b, KX, KZ, K2, N, dx, dz)
            if (n + 1) in passos_snapshot:
                _, _, b_f, _ = campos_fisicos(zeta, b, KX, KZ, K2)
                snapshots[n + 1] = b_f.copy()

    elif nome_esquema == "Adams-Bashforth-2":
        dzeta_ant, db_ant = inicializar_ab2_espectral(zeta_hat0, b_hat0, KX, K2, N, dt)
        zeta, b = zeta_hat0.copy(), b_hat0.copy()
        energias[0] = energia_fisica(zeta, b, KX, KZ, K2, N, dx, dz)
        _, _, b_f, _ = campos_fisicos(zeta, b, KX, KZ, K2)
        if 0 in passos_snapshot:
            snapshots[0] = b_f.copy()
        for n in range(n_passos):
            zeta, b, dzeta_ant, db_ant = adams_bashforth2_espectral(
                zeta, b, dzeta_ant, db_ant, KX, K2, N, dt
            )
            energias[n + 1] = energia_fisica(zeta, b, KX, KZ, K2, N, dx, dz)
            if (n + 1) in passos_snapshot:
                _, _, b_f, _ = campos_fisicos(zeta, b, KX, KZ, K2)
                snapshots[n + 1] = b_f.copy()

    elif nome_esquema == "Leapfrog":
        zeta_m1, b_m1 = inicializar_leapfrog_espectral(zeta_hat0, b_hat0, KX, K2, N, dt)
        zeta_ant, b_ant = zeta_m1, b_m1
        zeta_atu, b_atu = zeta_hat0.copy(), b_hat0.copy()
        energias[0] = energia_fisica(zeta_atu, b_atu, KX, KZ, K2, N, dx, dz)
        _, _, b_f, _ = campos_fisicos(zeta_atu, b_atu, KX, KZ, K2)
        if 0 in passos_snapshot:
            snapshots[0] = b_f.copy()
        for n in range(n_passos):
            zeta_novo, b_novo = leapfrog_espectral(zeta_ant, b_ant, zeta_atu, b_atu, KX, K2, N, dt)
            zeta_ant, b_ant = zeta_atu, b_atu
            zeta_atu, b_atu = zeta_novo, b_novo
            energias[n + 1] = energia_fisica(zeta_atu, b_atu, KX, KZ, K2, N, dx, dz)
            if (n + 1) in passos_snapshot:
                _, _, b_f, _ = campos_fisicos(zeta_atu, b_atu, KX, KZ, K2)
                snapshots[n + 1] = b_f.copy()

    elif nome_esquema == "Leapfrog (RAF)":
        # Mesma integracao Leapfrog acima, mas agora com o filtro de
        # Asselin aplicado A CADA PASSO (alpha=0.05, valor operacional
        # tipico) -- esta variante estava DEFINIDA no modulo
        # oceano_2d_schemes.py (robert_asselin_filtro) mas nunca era
        # de fato chamada em nenhum script da extensao 2D. Adicionada
        # aqui para fechar essa lacuna e mostrar, no contexto 2D
        # (espectro largo de modos, pacote de onda realista), o mesmo
        # compromisso supressao-de-ruido vs. amortecimento-do-sinal ja
        # visto no modelo 0D. Logica identica a de
        # oceano_coluna_energia.py (integrar_leapfrog com alpha!=None):
        # o argumento "ant" do Leapfrog usa sempre o nivel FILTRADO,
        # o argumento "atu" usa sempre o nivel BRUTO (nao filtrado).
        alpha_raf = 0.05
        zeta_filt_ant, b_filt_ant = inicializar_leapfrog_espectral(
            zeta_hat0, b_hat0, KX, K2, N, dt
        )
        zeta_raw_atu, b_raw_atu = zeta_hat0.copy(), b_hat0.copy()
        energias[0] = energia_fisica(zeta_raw_atu, b_raw_atu, KX, KZ, K2, N, dx, dz)
        _, _, b_f, _ = campos_fisicos(zeta_raw_atu, b_raw_atu, KX, KZ, K2)
        if 0 in passos_snapshot:
            snapshots[0] = b_f.copy()

        for n in range(n_passos):
            zeta_novo, b_novo = leapfrog_espectral(
                zeta_filt_ant, b_filt_ant, zeta_raw_atu, b_raw_atu, KX, K2, N, dt
            )
            zeta_filt_atu = robert_asselin_filtro(zeta_raw_atu, zeta_novo, zeta_filt_ant, alpha_raf)
            b_filt_atu = robert_asselin_filtro(b_raw_atu, b_novo, b_filt_ant, alpha_raf)

            energias[n + 1] = energia_fisica(zeta_novo, b_novo, KX, KZ, K2, N, dx, dz)
            if (n + 1) in passos_snapshot:
                _, _, b_f, _ = campos_fisicos(zeta_novo, b_novo, KX, KZ, K2)
                snapshots[n + 1] = b_f.copy()

            zeta_filt_ant, b_filt_ant = zeta_filt_atu, b_filt_atu
            zeta_raw_atu, b_raw_atu = zeta_novo, b_novo

    return energias, snapshots


ESQUEMAS = ["RK4", "Adams-Bashforth-2", "Leapfrog", "Leapfrog (RAF)"]


energias = {}
snapshots = {}
for nome in ESQUEMAS:
    e, s = integrar(nome)
    energias[nome] = e
    snapshots[nome] = s
    print(f"{nome:20s}: E(t_fim)/E(0) = {e[-1]/E0:.4f}")

# ---------------------------------------------------------------------
# Figura 1: snapshots do campo de empuxo em varios instantes, para
# cada esquema (linhas) x instante (colunas)
# ---------------------------------------------------------------------
n_col = len(passos_snapshot)
n_lin = len(ESQUEMAS)
fig, axs = plt.subplots(n_lin, n_col, figsize=(3.2 * n_col, 2.7 * n_lin), sharex=True, sharey=True)

vmax_b = 0.011
for i, nome in enumerate(ESQUEMAS):
    for j, passo in enumerate(passos_snapshot):
        ax = axs[i, j]
        campo = snapshots[nome][passo]
        pc = ax.pcolormesh(x / 1000.0, z, campo, shading="auto", cmap="RdBu_r",
                            vmin=-vmax_b, vmax=vmax_b)
        if i == 0:
            ax.set_title(f"t = {passo*dt/3600.0:.2f} h", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{nome}\nz (m)", fontsize=9)
        if i == n_lin - 1:
            ax.set_xlabel("x (km)")

fig.suptitle(
    "MET-576-4 - Propagacao/dispersao de um pacote de onda interna de empuxo\n"
    "Campo b(x,z,t): RK4 (referencia) vs. AB2 vs. Leapfrog vs. Leapfrog+RAF",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("mo576_2d_pacote_snapshots.png", dpi=150)
print("Figura salva: mo576_2d_pacote_snapshots.png")
plt.close(fig)

# ---------------------------------------------------------------------
# Figura 2: energia total do pacote ao longo do tempo, cada esquema
# ---------------------------------------------------------------------
t_h = np.arange(n_passos + 1) * dt / 3600.0

fig, ax = plt.subplots(figsize=(9, 5.5))
for nome in ESQUEMAS:
    ax.plot(t_h, energias[nome] / E0, linewidth=1.4, label=nome)
ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="Exato (conservado)")
ax.set_xlabel("Tempo (horas)")
ax.set_ylabel("E(t) / E(0)")
ax.set_title(
    "MET-576-4 - Energia do pacote de onda interna 2D\n"
    f"(dt = {dt:.0f} s, {n_passos} passos, RAF com alpha=0.05)"
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("mo576_2d_pacote_energia.png", dpi=150)
print("Figura salva: mo576_2d_pacote_energia.png")
plt.close(fig)
