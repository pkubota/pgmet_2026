# -*- coding: utf-8 -*-
"""
MET-576-4 - Extensao 2D (nivel x longitude) do modelo da coluna oceanica
Ondas internas de gravidade, equacoes de Boussinesq linearizadas em 2D
(x = longitude/horizontal, z = nivel/vertical), dominio periodico nas
duas direcoes, resolvido por metodo PSEUDO-ESPECTRAL (FFT dupla).

Equacoes (sem rotacao, para manter o foco no problema de integracao
temporal; a extensao com f e um exercicio natural indicado no relatorio):

    du/dt = -dphi/dx
    dw/dt = -dphi/dz + b
    db/dt = -N^2 w
    du/dx + dw/dz = 0                 (continuidade, Boussinesq)

Formulacao vorticidade-funcao de corrente (elimina phi e a restricao de
continuidade automaticamente):

    zeta = dw/dx - du/dz = laplaciano(psi),   u = -dpsi/dz,  w = dpsi/dx

    dzeta/dt = db/dx
    db/dt   = -N^2 w = -N^2 dpsi/dx

Em espaco de Fourier (dominio periodico Lx x Lz, numeros de onda
kx, kz), cada modo (kx, kz) desacopla dos demais (o sistema e' LINEAR)
e evolui exatamente como o oscilador 0D ja estudado no trabalho, mas
com frequencia dada pela RELACAO DE DISPERSAO DE ONDAS INTERNAS DE
GRAVIDADE:

    omega(kx, kz) = N * |kx| / sqrt(kx^2 + kz^2) = N * sin(theta)

onde theta e' o angulo entre o vetor de onda e a vertical. O caso 0D
anterior (oscilador de empuxo puro) e' exatamente o limite kz -> 0
(vetor de onda puramente horizontal): omega -> N, que e' precisamente a
frequencia usada em oceano_coluna_schemes.py. A extensao 2D preserva
TODOS os resultados anteriores como um caso particular e adiciona a
dependencia angular da frequencia, que e' o que introduz o interesse
numerico novo: esquemas que pareciam bem comportados para omega = N
podem se comportar de forma bem diferente para modos com omega << N
(vetores de onda quase verticais).

Prognosticos: zeta_hat(kx,kz,t), b_hat(kx,kz,t) (campos complexos no
espaco espectral). Diagnostico: psi_hat = -zeta_hat / (kx^2+kz^2)
(inversao do Laplaciano, trivial em espaco de Fourier); u, w fisicos
por derivacao espectral de psi.

Convencao de eixos: arrays tem forma (nz, nx) -- eixo 0 = nivel
(vertical), eixo 1 = longitude (horizontal) -- espelhando a convencao
usual de dados atmosfericos/oceanicos (lev, lon).
"""

import numpy as np


# ---------------------------------------------------------------------
# Grade e numeros de onda
# ---------------------------------------------------------------------

def criar_grade(nx, nz, Lx, Lz):
    """
    Cria a grade fisica (x, z) e os numeros de onda (KX, KZ) associados
    a um dominio periodico Lx x Lz discretizado em nx x nz pontos.

    Retorna: x, z, dx, dz, KX, KZ, K2
      KX, KZ, K2 tem forma (nz, nx) -- eixo 0 = nivel, eixo 1 = longitude.
    """
    dx = Lx / nx
    dz = Lz / nz
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz

    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    kz_1d = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)

    KZ, KX = np.meshgrid(kz_1d, kx_1d, indexing="ij")  # forma (nz, nx)
    K2 = KX ** 2 + KZ ** 2
    return x, z, dx, dz, KX, KZ, K2


def omega_dispersao(KX, K2, N):
    """
    Relacao de dispersao de ondas internas de gravidade:
        omega(kx,kz) = N |kx| / sqrt(kx^2+kz^2)

    O modo (kx=0, kz=0) (media do dominio) e qualquer modo com kx=0
    (vetor de onda puramente vertical) tem omega = 0 -- sao modos
    ESTACIONARIOS (nao oscilam: fisicamente, uma perturbacao de empuxo
    que so varia em z, sem variacao horizontal, nao gera movimento
    algum neste sistema linearizado sem rotacao).
    """
    K2_seguro = np.where(K2 > 0, K2, 1.0)
    omega = N * np.abs(KX) / np.sqrt(K2_seguro)
    omega = np.where(K2 > 0, omega, 0.0)
    return omega


# ---------------------------------------------------------------------
# Operador tangente (lado direito do sistema, em espaco espectral)
# ---------------------------------------------------------------------

def derivada_espectral(zeta_hat, b_hat, KX, K2, N):
    """
    dzeta_hat/dt = i kx b_hat
    db_hat/dt   = -N^2 w_hat = -N^2 (i kx psi_hat) = i N^2 kx zeta_hat / K2

    (usa psi_hat = -zeta_hat/K2, com o modo K2=0 mantido em zero --
    esse modo corresponde ao escoamento medio, sem dinamica de onda).
    """
    K2_seguro = np.where(K2 > 0, K2, 1.0)
    psi_hat = np.where(K2 > 0, -zeta_hat / K2_seguro, 0.0)
    w_hat = 1j * KX * psi_hat

    dzeta_hat_dt = 1j * KX * b_hat
    db_hat_dt = -(N ** 2) * w_hat
    return dzeta_hat_dt, db_hat_dt


# ---------------------------------------------------------------------
# Solucao analitica por modo (usada como referencia de verificacao)
# ---------------------------------------------------------------------

def solucao_analitica_espectral(t, zeta_hat0, b_hat0, KX, K2, N):
    """
    Cada modo de Fourier evolui como um oscilador harmonico de
    frequencia omega(kx,kz) (ver derivacao no cabecalho do modulo, via
    diagonalizacao da matriz 2x2 do sistema linear). Solucao fechada:

        zeta_hat(t) = cos(omega t) zeta_hat0
                      + sin(omega t)/omega * (i kx b_hat0)
        b_hat(t)    = cos(omega t) b_hat0
                      + sin(omega t)/omega * (i N^2 kx / K2 * zeta_hat0)

    Nos modos estacionarios (omega=0: kx=0 ou K2=0), a solucao e'
    constante no tempo (zeta_hat0, b_hat0), tratado por limite direto.
    """
    omega = omega_dispersao(KX, K2, N)
    omega_seguro = np.where(omega > 0, omega, 1.0)
    K2_seguro = np.where(K2 > 0, K2, 1.0)

    sinc_termo = np.where(omega > 0, np.sin(omega * t) / omega_seguro, 0.0)
    cos_termo = np.cos(omega * t)

    zeta_hat = cos_termo * zeta_hat0 + sinc_termo * (1j * KX * b_hat0)
    b_hat = cos_termo * b_hat0 + sinc_termo * (1j * (N ** 2) * KX / K2_seguro * zeta_hat0)

    # modos estacionarios: forcar exatamente o valor inicial (evita
    # ruido de ponto flutuante nas divisoes por K2/omega "seguros")
    estacionario = (KX == 0) | (K2 == 0)
    zeta_hat = np.where(estacionario, zeta_hat0, zeta_hat)
    b_hat = np.where(estacionario, b_hat0, b_hat)
    return zeta_hat, b_hat


# ---------------------------------------------------------------------
# Esquemas de integracao temporal (identicos em forma aos de
# oceano_coluna_schemes.py, agora operando sobre arrays 2D complexos --
# cada modo de Fourier e' avancado em paralelo, elementwise)
# ---------------------------------------------------------------------

def rk4_espectral(zeta_hat, b_hat, KX, K2, N, dt):
    k1_z, k1_b = derivada_espectral(zeta_hat, b_hat, KX, K2, N)
    k2_z, k2_b = derivada_espectral(zeta_hat + 0.5 * dt * k1_z, b_hat + 0.5 * dt * k1_b, KX, K2, N)
    k3_z, k3_b = derivada_espectral(zeta_hat + 0.5 * dt * k2_z, b_hat + 0.5 * dt * k2_b, KX, K2, N)
    k4_z, k4_b = derivada_espectral(zeta_hat + dt * k3_z, b_hat + dt * k3_b, KX, K2, N)

    zeta_novo = zeta_hat + (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
    b_novo = b_hat + (dt / 6.0) * (k1_b + 2 * k2_b + 2 * k3_b + k4_b)
    return zeta_novo, b_novo


def adams_bashforth2_espectral(zeta_atu, b_atu, dzeta_ant, db_ant, KX, K2, N, dt):
    dzeta_atu, db_atu = derivada_espectral(zeta_atu, b_atu, KX, K2, N)
    zeta_novo = zeta_atu + (dt / 2.0) * (3.0 * dzeta_atu - dzeta_ant)
    b_novo = b_atu + (dt / 2.0) * (3.0 * db_atu - db_ant)
    return zeta_novo, b_novo, dzeta_atu, db_atu


def inicializar_ab2_espectral(zeta_hat0, b_hat0, KX, K2, N, dt):
    """Um passo de RK4 para tras no tempo, para obter f_(-1)."""
    zeta_m1, b_m1 = rk4_espectral(zeta_hat0, b_hat0, KX, K2, N, -dt)
    return derivada_espectral(zeta_m1, b_m1, KX, K2, N)


def leapfrog_espectral(zeta_ant, b_ant, zeta_atu, b_atu, KX, K2, N, dt):
    dzeta_atu, db_atu = derivada_espectral(zeta_atu, b_atu, KX, K2, N)
    zeta_novo = zeta_ant + 2.0 * dt * dzeta_atu
    b_novo = b_ant + 2.0 * dt * db_atu
    return zeta_novo, b_novo


def inicializar_leapfrog_espectral(zeta_hat0, b_hat0, KX, K2, N, dt):
    """Um passo de RK4 para tras no tempo, para obter o nivel n=-1."""
    return rk4_espectral(zeta_hat0, b_hat0, KX, K2, N, -dt)


def robert_asselin_filtro(y_atu, y_novo, y_atu_filtrado_ant, alpha=0.1):
    """Identico ao filtro usado no modelo 0D (formula independe de
    dimensionalidade: opera elementwise, funciona igual em arrays 2D)."""
    return y_atu + alpha * (y_atu_filtrado_ant - 2.0 * y_atu + y_novo)


# ---------------------------------------------------------------------
# Reconstrucao de campos fisicos e diagnostico de energia
# ---------------------------------------------------------------------

def campos_fisicos(zeta_hat, b_hat, KX, KZ, K2):
    """
    Retorna os campos fisicos (u, w, b, zeta) no espaco real (nz, nx),
    a partir dos campos espectrais prognosticos (zeta_hat, b_hat).
    """
    K2_seguro = np.where(K2 > 0, K2, 1.0)
    psi_hat = np.where(K2 > 0, -zeta_hat / K2_seguro, 0.0)
    u_hat = -1j * KZ * psi_hat
    w_hat = 1j * KX * psi_hat

    u = np.real(np.fft.ifft2(u_hat))
    w = np.real(np.fft.ifft2(w_hat))
    b = np.real(np.fft.ifft2(b_hat))
    zeta = np.real(np.fft.ifft2(zeta_hat))
    return u, w, b, zeta


def energia_fisica(zeta_hat, b_hat, KX, KZ, K2, N, dx, dz):
    """
    Energia mecanica total (cinetica + potencial disponivel), integrada
    sobre o dominio fisico:
        E = integral 1/2 (u^2 + w^2 + b^2/N^2) dx dz
          ~ sum_grade 1/2 (u^2+w^2+b^2/N^2) * dx * dz
    """
    u, w, b, _ = campos_fisicos(zeta_hat, b_hat, KX, KZ, K2)
    densidade = 0.5 * (u ** 2 + w ** 2 + (b ** 2) / (N ** 2))
    return np.sum(densidade) * dx * dz
