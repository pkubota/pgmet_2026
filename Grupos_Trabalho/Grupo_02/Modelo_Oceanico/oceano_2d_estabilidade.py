# -*- coding: utf-8 -*-
"""
MET-576-4 - Extensao 2D: mapa de estabilidade sobre o espectro (kx, kz)

Diagnostico NOVO, so possivel gracas a extensao 2D: para um Delta t
fixo (escolha unica de passo de tempo do modelo), a relacao de
dispersao omega(kx,kz) = N|kx|/sqrt(kx^2+kz^2) faz com que MODOS
DIFERENTES do dominio sejam integrados com graus MUITO diferentes de
precisao pelo mesmo esquema numerico -- pois cada modo "ve" um x =
omega(kx,kz)*dt diferente, e o fator de amplificacao lambda(x) de cada
esquema (calculado em oceano_coluna_estabilidade.py) depende de x.

Em particular:
  - Modos com vetor de onda quase-horizontal (kz pequeno, kx grande):
    omega proximo de N (o maximo possivel) -- x = N dt e' o "pior caso"
    ja estudado na versao 0D do trabalho.
  - Modos com vetor de onda quase-vertical (kx pequeno, kz grande):
    omega proximo de 0 -- x proximo de 0, excelente precisao para
    QUALQUER esquema (inclusive o Euler explicito, que e' instavel na
    versao 0D mas seria "aceitavel" nesses modos quase estacionarios).
  - Ou seja, o mesmo esquema, com o mesmo dt, pode ser excelente para
    alguns modos do dominio e ruim para outros -- um fenomeno que a
    versao 0D (que so testa omega=N) nao consegue revelar.

Saida: mo576_2d_mapa_estabilidade.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oceano_2d_schemes import criar_grade, omega_dispersao


# ---------------------------------------------------------------------
# Formulas do fator de amplificacao (identicas as de
# oceano_coluna_estabilidade.py, repetidas aqui para manter este script
# autocontido)
# ---------------------------------------------------------------------

def lambda_rk4(x):
    z = -1j * x
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0


def lambda_leapfrog_fisico(x):
    disc = np.asarray(1.0 - x ** 2, dtype=complex)
    return -1j * x + np.sqrt(disc)


def lambda_ab2_fisico(x):
    """
    Raiz fisica da equacao lambda^2 - (1+1.5z)lambda + 0.5z = 0, z=-ix.
    Identificada por continuidade a partir de x=0 (lambda_fis -> 1).
    """
    x_flat = x.ravel()
    z = -1j * x_flat
    coef_b = -(1.0 + 1.5 * z)
    coef_c = 0.5 * z
    disc = np.asarray(coef_b ** 2 - 4 * coef_c, dtype=complex)
    raiz = np.sqrt(disc)
    lam1 = (-coef_b + raiz) / 2.0
    lam2 = (-coef_b - raiz) / 2.0

    ordem = np.argsort(x_flat)
    fis_ordenado = np.empty(len(x_flat), dtype=complex)
    lam1_o, lam2_o = lam1[ordem], lam2[ordem]
    fis_ordenado[0] = lam1_o[0] if abs(lam1_o[0] - 1.0) < abs(lam2_o[0] - 1.0) else lam2_o[0]
    for i in range(1, len(ordem)):
        d1 = abs(lam1_o[i] - fis_ordenado[i - 1])
        d2 = abs(lam2_o[i] - fis_ordenado[i - 1])
        fis_ordenado[i] = lam1_o[i] if d1 <= d2 else lam2_o[i]

    fis = np.empty(len(x_flat), dtype=complex)
    fis[ordem] = fis_ordenado
    return fis.reshape(x.shape)


# ---------------------------------------------------------------------
# Grade e relacao de dispersao
# ---------------------------------------------------------------------
nx, nz = 64, 64
Lx, Lz = 200000.0, 2000.0  # 200 km x 2 km -- razao de aspecto tipica p/ ondas internas
N = 0.01
dt = 60.0  # mesmo dt usado na simulacao de energia (60 dias) do modelo 0D->2D

x, z, dx, dz, KX, KZ, K2 = criar_grade(nx, nz, Lx, Lz)
omega = omega_dispersao(KX, K2, N)
X = omega * dt  # parametro adimensional de estabilidade, um valor por modo

# reorganiza para plotagem com kx, kz crescentes e centrados (fftshift)
KX_plot = np.fft.fftshift(KX, axes=1)
KZ_plot = np.fft.fftshift(KZ, axes=0)
X_plot = np.fft.fftshift(np.fft.fftshift(X, axes=0), axes=1)

mod_lam_rk4 = np.abs(lambda_rk4(X_plot))
mod_lam_lf = np.abs(lambda_leapfrog_fisico(X_plot))
mod_lam_ab2 = np.abs(lambda_ab2_fisico(X_plot))

# ---------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 11))

ax = axs[0, 0]
pc = ax.pcolormesh(KX_plot * 1000, KZ_plot * 1000, X_plot, shading="auto", cmap="viridis")
ax.set_xlabel("kx (rad/km)")
ax.set_ylabel("kz (rad/km)")
ax.set_title(f"(a) x = omega(kx,kz) * dt\n(dt = {dt:.0f} s)")
fig.colorbar(pc, ax=ax, label="x = omega dt")

ax = axs[0, 1]
pc = ax.pcolormesh(KX_plot * 1000, KZ_plot * 1000, mod_lam_rk4, shading="auto",
                    cmap="RdBu_r", vmin=0.9995, vmax=1.0005)
ax.set_xlabel("kx (rad/km)")
ax.set_ylabel("kz (rad/km)")
ax.set_title("(b) |lambda| RK4: quase 1 no espectro todo\n(fraca dissipacao mesmo perto de omega=N)")
fig.colorbar(pc, ax=ax, label="|lambda|")

ax = axs[1, 0]
pc = ax.pcolormesh(KX_plot * 1000, KZ_plot * 1000, mod_lam_lf, shading="auto",
                    cmap="RdBu_r", vmin=0.9, vmax=1.1)
ax.set_xlabel("kx (rad/km)")
ax.set_ylabel("kz (rad/km)")
ax.set_title("(c) |lambda| Leapfrog (modo fisico) -- neutro (=1)\nem TODO o espectro resolvido")
fig.colorbar(pc, ax=ax, label="|lambda|")

ax = axs[1, 1]
pc = ax.pcolormesh(KX_plot * 1000, KZ_plot * 1000, mod_lam_ab2, shading="auto",
                    cmap="RdBu_r", vmin=1.0, vmax=1.07)
ax.set_xlabel("kx (rad/km)")
ax.set_ylabel("kz (rad/km)")
ax.set_title("(d) |lambda| AB2 -- amplifica mais forte exatamente\nnos modos quase-horizontais (kz pequeno, omega ~ N)")
fig.colorbar(pc, ax=ax, label="|lambda|")

fig.suptitle(
    "MET-576-4 - Mapa de estabilidade sobre o espectro (kx,kz): mesmo dt,\n"
    "graus de precisao MUITO diferentes conforme o angulo de propagacao da onda",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("mo576_2d_mapa_estabilidade.png", dpi=150)
print("Figura salva: mo576_2d_mapa_estabilidade.png")

print()
print(f"Faixa de x = omega*dt no dominio (dt={dt:.0f}s): "
      f"[{X.min():.4f}, {X.max():.4f}] (max = N*dt = {N*dt:.4f})")
print("Modos com kz=0 (linha horizontal central do mapa) atingem exatamente x = N dt,")
print("o pior caso ja estudado na versao 0D do trabalho.")
