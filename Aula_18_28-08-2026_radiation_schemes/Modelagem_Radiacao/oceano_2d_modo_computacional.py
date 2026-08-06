# -*- coding: utf-8 -*-
"""
MET-576-4 - Extensao 2D: modo computacional do Leapfrog no pacote de
onda interna (fecha a lacuna do filtro de Asselin nunca ter sido
efetivamente excitado/testado nos scripts anteriores da extensao 2D)

Mesma estrategia de demonstracao usada no modelo 0D
(oceano_coluna_modo_computacional.py): o nivel de tempo n-1 usado para
iniciar o Leapfrog recebe uma pequena perturbacao artificial ALEM da
inicializacao por RK4, projetando uma componente finita sobre o modo
computacional logo no primeiro passo. Isso e feito aqui em campo 2D
completo (nao apenas em um unico modo de Fourier), deixando visivel a
assinatura espacial do ruido "2 dt": um padrao de tabuleiro de xadrez
que se sobrepoe ao campo fisico do pacote de onda.

Saidas:
  mo576_2d_modo_computacional.png  - snapshots com e sem a perturbacao
                                      artificial, revelando o ruido 2dt
  mo576_2d_asselin_alpha.png       - energia do pacote perturbado para
                                      varios valores de alpha do RAF
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oceano_2d_schemes import (
    criar_grade, rk4_espectral, leapfrog_espectral,
    inicializar_leapfrog_espectral, robert_asselin_filtro,
    campos_fisicos, energia_fisica,
)

# ---------------------------------------------------------------------
# Grade e parametros -- identicos ao script do pacote de onda, para
# permitir comparacao direta
# ---------------------------------------------------------------------
nx, nz = 128, 64
Lx, Lz = 200000.0, 2000.0
N = 0.01
dt = 20.0
n_passos = 200  # janela mais curta que o pacote de onda principal,
                 # o suficiente para deixar o ruido 2dt bem visivel
                 # sem que ele va dominando demais o campo fisico

x, z, dx, dz, KX, KZ, K2 = criar_grade(nx, nz, Lx, Lz)

x0, z0 = Lx / 2.0, Lz / 2.0
sigma_x, sigma_z = 8000.0, 150.0

X2, Z2 = np.meshgrid(x, z)
b_inicial = 0.01 * np.exp(-((X2 - x0) ** 2) / (2 * sigma_x ** 2)
                           - ((Z2 - z0) ** 2) / (2 * sigma_z ** 2))
zeta_inicial = np.zeros_like(b_inicial)

b_hat0 = np.fft.fft2(b_inicial)
zeta_hat0 = np.fft.fft2(zeta_inicial)
E0 = energia_fisica(zeta_hat0, b_hat0, KX, KZ, K2, N, dx, dz)

# Perturbacao artificial no nivel n-1: um segundo pacote gaussiano,
# mais fraco e levemente deslocado, somado ao campo de EMPUXO (b) --
# nao ao de vorticidade. Perturbar zeta diretamente e' problematico:
# a inversao de Poisson psi_hat=-zeta_hat/K2 AMPLIFICA fortemente
# qualquer perturbacao de zeta concentrada em numeros de onda pequenos
# (K2 pequeno), o que gera velocidades e energia irrealisticamente
# grandes (testado e descartado). Perturbar b evita esse problema
# porque a contribuicao de b para a energia e' direta (0.5 b^2/N^2,
# sem divisao por K2), exatamente como no modelo 0D.
amplitude_perturbacao = 0.15  # fracao da amplitude do pacote original
b_perturbacao = amplitude_perturbacao * 0.01 * np.exp(
    -((X2 - x0) ** 2) / (2 * (1.5 * sigma_x) ** 2)
    - ((Z2 - z0) ** 2) / (2 * (1.5 * sigma_z) ** 2)
)
b_hat_perturbacao = np.fft.fft2(b_perturbacao)


def integrar_leapfrog_2d_perturbado(alpha=None):
    """
    Leapfrog com o nivel n-1 perturbado (zeta_hat0 + perturbacao),
    com ou sem filtro de Asselin. Retorna lista de campos de b(x,z) em
    cada passo salvo e a serie de energia completa.
    """
    zeta_m1_exato, b_m1_exato = inicializar_leapfrog_espectral(
        zeta_hat0, b_hat0, KX, K2, N, dt
    )
    zeta_m1 = zeta_m1_exato
    b_m1 = b_m1_exato + b_hat_perturbacao

    zeta_filt_ant, b_filt_ant = zeta_m1, b_m1
    zeta_raw_atu, b_raw_atu = zeta_hat0.copy(), b_hat0.copy()

    energias = np.zeros(n_passos + 1)
    energias[0] = energia_fisica(zeta_raw_atu, b_raw_atu, KX, KZ, K2, N, dx, dz)
    snapshots = {0: campos_fisicos(zeta_raw_atu, b_raw_atu, KX, KZ, K2)[2]}

    passos_salvos = [0, 50, 100, 150, 200]

    for n in range(n_passos):
        if alpha is None:
            zeta_novo, b_novo = leapfrog_espectral(
                zeta_filt_ant, b_filt_ant, zeta_raw_atu, b_raw_atu, KX, K2, N, dt
            )
            zeta_filt_ant, b_filt_ant = zeta_raw_atu, b_raw_atu  # sem filtro: "ant" e' so o anterior bruto
        else:
            zeta_novo, b_novo = leapfrog_espectral(
                zeta_filt_ant, b_filt_ant, zeta_raw_atu, b_raw_atu, KX, K2, N, dt
            )
            zeta_filt_atu = robert_asselin_filtro(zeta_raw_atu, zeta_novo, zeta_filt_ant, alpha)
            b_filt_atu = robert_asselin_filtro(b_raw_atu, b_novo, b_filt_ant, alpha)
            zeta_filt_ant, b_filt_ant = zeta_filt_atu, b_filt_atu

        zeta_raw_atu, b_raw_atu = zeta_novo, b_novo
        energias[n + 1] = energia_fisica(zeta_novo, b_novo, KX, KZ, K2, N, dx, dz)
        if (n + 1) in passos_salvos:
            _, _, b_f, _ = campos_fisicos(zeta_novo, b_novo, KX, KZ, K2)
            snapshots[n + 1] = b_f

    return energias, snapshots


# ---------------------------------------------------------------------
# Figura 1: snapshots com e sem a perturbacao do modo computacional
# ---------------------------------------------------------------------
energias_bruto, snapshots_bruto = integrar_leapfrog_2d_perturbado(alpha=None)

# referencia sem perturbacao (RK4, "limpo") para comparacao visual
zeta_ref, b_ref = zeta_hat0.copy(), b_hat0.copy()
snapshots_ref = {0: campos_fisicos(zeta_ref, b_ref, KX, KZ, K2)[2]}
passos_salvos = [0, 50, 100, 150, 200]
for n in range(n_passos):
    zeta_ref, b_ref = rk4_espectral(zeta_ref, b_ref, KX, K2, N, dt)
    if (n + 1) in passos_salvos:
        _, _, b_f, _ = campos_fisicos(zeta_ref, b_ref, KX, KZ, K2)
        snapshots_ref[n + 1] = b_f

fig, axs = plt.subplots(2, len(passos_salvos), figsize=(3.2 * len(passos_salvos), 5.5),
                         sharex=True, sharey=True)
vmax_b = 0.011
for j, passo in enumerate(passos_salvos):
    ax = axs[0, j]
    pc = ax.pcolormesh(x / 1000.0, z, snapshots_ref[passo], shading="auto",
                        cmap="RdBu_r", vmin=-vmax_b, vmax=vmax_b)
    ax.set_title(f"t = {passo*dt/3600.0:.2f} h", fontsize=10)
    if j == 0:
        ax.set_ylabel("RK4 (referencia,\nsem perturbacao)\nz (m)", fontsize=9)

    ax = axs[1, j]
    pc = ax.pcolormesh(x / 1000.0, z, snapshots_bruto[passo], shading="auto",
                        cmap="RdBu_r", vmin=-vmax_b, vmax=vmax_b)
    ax.set_xlabel("x (km)")
    if j == 0:
        ax.set_ylabel("Leapfrog (n-1\nperturbado, SEM RAF)\nz (m)", fontsize=9)

fig.suptitle(
    "MET-576-4 - Modo computacional do Leapfrog em campo 2D:\n"
    "perturbacao no nivel n-1 gera ruido \"2 dt\" sobreposto ao pacote fisico",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("mo576_2d_modo_computacional.png", dpi=150)
print("Figura salva: mo576_2d_modo_computacional.png")
plt.close(fig)

# ---------------------------------------------------------------------
# Figura 2: energia do pacote perturbado para varios alpha do RAF
# ---------------------------------------------------------------------
alphas = [0.0, 0.01, 0.05, 0.1, 0.2]
t_h = np.arange(n_passos + 1) * dt / 3600.0

fig, ax = plt.subplots(figsize=(9, 5.5))
for a in alphas:
    alpha_arg = None if a == 0.0 else a
    energias_a, _ = integrar_leapfrog_2d_perturbado(alpha=alpha_arg)
    rotulo = "sem filtro (alpha=0)" if a == 0.0 else f"alpha = {a}"
    ax.plot(t_h, energias_a / E0, linewidth=1.3, label=rotulo)
ax.axhline(1.0, color="k", linestyle=":", linewidth=1)
ax.set_xlabel("Tempo (horas)")
ax.set_ylabel("E(t) / E(0)")
ax.set_title(
    "MET-576-4 - Efeito do filtro de Asselin sobre o pacote 2D\n"
    "com o modo computacional excitado (nivel n-1 perturbado)"
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("mo576_2d_asselin_alpha.png", dpi=150)
print("Figura salva: mo576_2d_asselin_alpha.png")
plt.close(fig)

print()
print("Resumo (energia relativa ao final, campo 2D com n-1 perturbado):")
for a in alphas:
    alpha_arg = None if a == 0.0 else a
    energias_a, _ = integrar_leapfrog_2d_perturbado(alpha=alpha_arg)
    rotulo = "sem filtro" if a == 0.0 else f"alpha={a}"
    print(f"  {rotulo:14s}: E/E0 = {energias_a[-1]/E0:.4f}")
